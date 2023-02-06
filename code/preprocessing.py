import click
import pickle
import wfdb
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import utils

def load_and_dump_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.pkl', 'wb'), protocol=4)
    elif sampling_rate == 500:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.pkl', 'wb'), protocol=4)


def compute_label_aggregations(df, folder, ctype):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df

def compute_multihot_encodings(df, folder):
    
    def multihot_encode(labels, label_mapping, num_classes):
        label_ids = [np.where(label_mapping == label)[0] for label in labels]
        res = np.zeros(num_classes, dtype=np.float32)
        for label_id in label_ids:
            res[label_id] = 1
        return res

    for ctype in utils.label_mappings.keys():
        df = compute_label_aggregations(df, folder, ctype)
        label_mapping = np.array(utils.label_mappings[ctype])
        df[ctype+'_multihot'] = df[ctype].apply(
                lambda labels: multihot_encode(labels, label_mapping, len(label_mapping)))
        pickle.dump(np.vstack(df[ctype+'_multihot'].values).astype(int), open(os.path.join(folder, "multihot_"+str(ctype)+".npy"), 'wb'))
     
    return df

@click.command()
@click.option('--data_path', default='data/ptbxl/', help='path to dataset')
@click.option('--addon_path', default='addons/', help='path to addons')
@click.option('--sampling_rate', default=100, help='sampling rate')
def preprocessing(data_path, addon_path, sampling_rate):

    # load and convert annotation data
    df = pd.read_csv(data_path+'ptbxl_database.csv', index_col='ecg_id')

    # load and merge r-peaks 
    rpeaks_df = pd.read_csv(addon_path+'r-peaks.csv').set_index('ecg_id')
    df = df.merge(rpeaks_df, left_index=True, right_index=True)
    if sampling_rate == 100:
        upscale_r = 1
    else:
        upscale_r = 5
    df.r_peaks = df.r_peaks.apply(lambda x: np.array(' '.join(x.split()).replace('[ ', '').replace('[','').replace(']','').replace(' ]','').split()).astype(int)*upscale_r)        

    # load and merge concepts
    concept_df = pd.read_csv(addon_path+'concepts.csv', index_col=0).set_index('ecg_id')
    df = df.merge(concept_df, left_index=True, right_index=True)
   
    # compute mutlihot encoding and store them in data_path
    df.scp_codes = df.scp_codes.apply(lambda x: eval(x))
    df = compute_multihot_encodings(df, data_path)

    # save enriched database for future usage
    df.to_csv(data_path+'ptbxl_database_enriched.csv')

    # load and dump raw data
    load_and_dump_raw_data_ptbxl(df,sampling_rate, data_path)
    
if __name__ == '__main__':
    preprocessing()