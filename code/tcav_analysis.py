import click
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

import shutil
from captum.concept import TCAV
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept import Concept
import utils
import ecg_model as m
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

mapping_associated_pathos = {
    'RS-LVH':['LVH'], 
    'S12-LVH':['LVH'], 
    'R56-LVH':['LVH'], 
    'QRS-LVH':['LVH'], 
    'LI-LVH':['LVH'], 
    'SLI-LVH':['LVH'], 
    'QRS-CLBB':['CLBBB', 'ILBBB'], 
    'ST-ELEV-MI':['IMI', 'AMI'], 
    'ST-DEPR-MI':['IMI', 'AMI'],
    'MI-ALL':['IMI', 'AMI']
}

subdiag_ordering = np.array([
    'NORM',    
    'LVH','RVH','SEHYP','LAO/LAE','RAO/RAE',
    'CLBBB','CRBBB','ILBBB','IRBBB','IVCD','LAFB/LPFB','WPW','_AVB',
    'AMI','IMI','LMI','PMI',   
    'ISCA','ISCI','ISC_','NST_','STTC'
])

def plot_aggregated_mean_tcav(result_df, concept, pathos, ordering=subdiag_ordering, save_path=None):

    plt.figure(figsize=(10,10))
    ax = plt.gca()
    im = ax.imshow(result_df[ordering].values, vmin=0, vmax=1, cmap='bwr')
    
    plt.xticks(range(len(ordering)), ordering, rotation=90)
    plt.yticks(range(len(result_df)), [ result_df.index[i] + ' [acc='+ result_df['cav-acc'].values[i] + ']' for i in range(len(result_df))])
    plt.title(concept)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im, cax=cax)

    for patho in pathos:
        pidx = np.argwhere(ordering == patho).flatten()[0]
        rect = patches.Rectangle((pidx-.5, -.5), 1, len(result_df)+1, linewidth=4, edgecolor='g', facecolor='none', zorder=10)
        ax.add_patch(rect)
        ax.get_xticklabels()[pidx].set_color('g')

    if save_path:
        plt.savefig(save_path+'TCAV_mean.jpg', bbox_inches='tight', pad_inches=.1)
    else:
        plt.show()

def plot_aggregated_std_tcav(result_df, concept, pathos, ordering=subdiag_ordering, save_path=None):

    plt.figure(figsize=(10,10))
    ax = plt.gca()
    im = ax.imshow(result_df[ordering].values, cmap='jet')
    plt.xticks(range(len(ordering)), ordering, rotation=90)
    plt.yticks(range(len(result_df)), [ result_df.index[i] + ' [acc='+ result_df['cav-acc'].values[i] + ']' for i in range(len(result_df))])
    plt.title(concept)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im, cax=cax)

    for patho in pathos:
        pidx = np.argwhere(ordering == patho).flatten()[0]
        rect = patches.Rectangle((pidx-.5, -.5), 1, len(result_df)+1, linewidth=4, edgecolor='g', facecolor='none', zorder=10)
        ax.add_patch(rect)
        ax.get_xticklabels()[pidx].set_color('g')

    if save_path:
        plt.savefig(save_path+'TCAV_std.jpg', bbox_inches='tight', pad_inches=.1)
    else:
        plt.show()

@click.command()
@click.option('--data_dir', default='data/ptbxl/', help='path to dataset')
@click.option('--feature_dir', default='ptbxl_addons/', help='path to feature and concept dataset')
@click.option('--modeltype', default='lenet', help='model type')
@click.option('--model_checkpoint_path', default=None, help='path to model checkpoint')
@click.option('--task', default='subdiagnostic', help='task')
@click.option('--logdir', default=None, help='log dir')
@click.option('--batch_size', default=64, help='batch_size')
@click.option('--n_fits', default=1, help='number of TCAV fits')
#@click.option('--concepts', default=['RS-LVH', 'S12-LVH', 'R56-LVH', 'QRS-LVH', 'LI-LVH', 'SLI-LVH','QRS-CLBB', 'ST-ELEV-MI', 'ST-DEPR-MI', 'MI-ALL'], help='list of relevant concepts')
@click.option('--concepts', default=['R56-LVH'], help='list of relevant concepts')
@click.option('--evaluate', default=True, is_flag=True)
@click.option('--force_retraining', default=True, is_flag=True)
def tcav_analysis(data_dir, feature_dir, modeltype, model_checkpoint_path, task, logdir, batch_size, n_fits, concepts, evaluate, force_retraining):
    # center crop of length 250
    n_from = 375
    n_to = 625
    
    # load dataset and dataframe with r-peaks and concepts
    signals,df,labels = utils.get_dataset_label(data_path=data_dir, task=task)
    signals = torch.from_numpy(np.swapaxes(signals,1,2).astype(np.float32))
    
    if modeltype == 'lenet':
        model = m.get_model('lenet',labels.shape[-1])
        model.load_from_checkpoint(model_checkpoint_path)
        model.eval()
        model = model.model
        model.to('cpu')

        layers=[str(i) for i in range(len(model.sequential))]
        layer_names = [s.split('(')[1].split(': ')[1] for s in str(model.sequential).split('\n')[1:-1]]   
    elif modeltype=='resnet':
        model = m.get_model('resnet',labels.shape[-1])
        model.load_from_checkpoint(model_checkpoint_path)
        model = model.model
        model.eval()
        modelchildren = list(model.children())
        
        modules = modelchildren[:-1]
        modules += [nn.Flatten()]
        modules += [modelchildren[-1]]
        model = nn.Sequential(*modules)
        model.to('cpu')
        layers=[str(i) for i in range(len(model))]
        layer_names = [str(type(x)).split(".")[-1].split("'")[0] for x in model]

    elif modeltype == 'xresnet':
        model = m.get_model('xresnet',labels.shape[-1])
        model.load_from_checkpoint(model_checkpoint_path)
        model = model.model
        model.eval()
        modelchildren1 = list(model.features.children()) # encoder
        modelchildren2 = list(model.children())[1:] # head
        modules = modelchildren1
        modules += [nn.Flatten()]
        modules += modelchildren2
        model = nn.Sequential(*modules)
        model.to('cpu')
        layers=[str(i) for i in range(len(model))]
        layer_names = [str(type(x)).split(".")[-1].split("'")[0] for x in model]

    if evaluate:
        # evaluate model
        y_test_pred = []
        samples = signals[df.strat_fold.values == 10]
        for i in tqdm(range(len(samples)//batch_size + 1)):
            yi = model.forward(samples[i*batch_size:(i+1)*batch_size].to('cpu').type(torch.float)).cpu().detach().numpy()
            y_test_pred.append(yi)
        y_test_pred = np.concatenate(y_test_pred)
        _ = utils.multiclass_roc_curve(labels[df.strat_fold == 10], y_test_pred, utils.label_mappings[task], title='ROC for ' + modeltype + ' on '+ task, savepath=logdir)

    for concept in eval(concepts):
        associated_pathos = mapping_associated_pathos[concept]

        cav_path = logdir+'cav/'+concept+'/'

        if force_retraining:
            try:
                shutil.rmtree(cav_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        tcav = TCAV(model=model.sequential,
            layers=layers,
            layer_attr_method=None,
            save_path=cav_path,
            )

        # create positive concept dataset
        positive_idxs = np.argwhere(df[concept].values).flatten()
        positive_samples = signals[positive_idxs, :,n_from:n_to]
        positive_concept = Concept(id=0, name=concept, data_iter=dataset_to_dataloader(positive_samples.to('cpu')))
        
        dfs = []
        for i in range(n_fits):
            # create ith random concept
            negative_idxs = np.random.choice(np.argwhere(~df[concept].values).flatten(), len(positive_idxs))
            negative_samples = signals[negative_idxs, :,n_from:n_to].to('cpu')
            negative_concept = Concept(id=i+1, name="random_"+str(i), data_iter=dataset_to_dataloader(negative_samples))
            concept_list = [[positive_concept, negative_concept]]

            # apply TCAV to each patho
            tcav_results_dic = {}
            for patho in utils.label_mappings[task]:
                cidx = np.argwhere(np.array(utils.label_mappings[task]) == patho).flatten()[0]
                patho_samples = signals[labels[:,cidx]==1, :,n_from:n_to]#.to('cpu')
                tcav_scores = tcav.interpret(
                    inputs=patho_samples,
                    experimental_sets=concept_list,
                    target=int(cidx),
                    processes=None,)
                patho_samples = patho_samples.cpu().detach()
                tcav_results_dic[patho] = tcav_scores
            negative_samples.cpu().detach()

            # delete activations to save space
            try:
                shutil.rmtree(cav_path+'av/')
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

            # convert to proper dataframes
            key = '0-'+str(i+1)
            dic = {'layer':layer_names,'cav-acc':[float(tcav.cavs[key][layer].stats['accs'].numpy()) for layer in layers]}
            for patho in utils.label_mappings[task]:
                dic[patho] = [tcav_results_dic[patho][key][layer]['sign_count'].cpu().numpy()[0] for layer in layers]
            tmp_df = pd.DataFrame(dic).set_index('layer')
            tmp_df.to_csv(cav_path+key+'.csv')
            dfs.append(tmp_df)

        # compute statistics of multiple fits
        mean_dic = {'layer':layer_names}
        std_dic = {'layer':layer_names}
            
        for col in tmp_df.columns:
            vals = np.array([dfi[col].values for dfi in dfs])
            mean_vals = np.mean(vals, axis=0)
            std_vals = np.std(vals, axis=0)
            if col == 'cav-acc':
                tmp_accs = [str(np.round(a[0],3))+'$\pm$'+str(np.round(a[1],3)) for a in zip(mean_vals, std_vals)]
                mean_dic[col] = tmp_accs
                std_dic[col] = tmp_accs
            else:
                mean_dic[col] = mean_vals
                std_dic[col] = std_vals
        mean_df = pd.DataFrame(mean_dic).set_index('layer')
        std_df = pd.DataFrame(std_dic).set_index('layer')

        # plot statistics of multiple fits
        plot_aggregated_mean_tcav(mean_df, 'mean TCAV scores for '+concept, associated_pathos, save_path=cav_path)
        plot_aggregated_std_tcav(std_df, 'std TCAV scores for '+concept, associated_pathos, save_path=cav_path)

if __name__ == '__main__':
    tcav_analysis()