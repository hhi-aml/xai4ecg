import os
import click
import random
import utils
import pandas as pd
import numpy as np
from pathlib import Path
from ecg_model import get_model
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class ECGDataModule(LightningDataModule):

    name = 'ecg_dataset'
    extra_args = {}

    def __init__(
            self,
            batch_size,
            data_dir, 
            addon_dir,
            label_class="all",
            num_workers: int = 8,
            data_input_size=250,
            shuffle_train=True,
            drop_last=True,
            val_fold=9,
            test_fold=10,
            lead_order=None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dims = (12, data_input_size)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.addon_dir = Path(addon_dir)
        self.label_class = label_class
        self.data_input_size = data_input_size
        self.shuffle_train = shuffle_train
        self.drop_last = drop_last
        self.val_fold = val_fold
        self.test_fold = test_fold
        train_folds = list(range(1, 11))
        train_folds.remove(test_fold)
        train_folds.remove(val_fold)
        self.train_folds = np.array(train_folds)
        self.lead_order = utils.leads
        self.set_params()

    def set_params(self):
        if self.label_class == 'T_Wave_Amplitude':
            self.dataset, self.labels, self.df = utils.get_regression_data(task=self.label_class)
            self.num_samples = len(self.dataset)
            self.num_classes = self.labels.shape[-1]

        elif self.label_class == 'P_Wave_Amplitude':
            self.dataset, self.labels, self.df = utils.get_regression_data(task=self.label_class)
            self.num_samples = len(self.dataset)
            self.num_classes = self.labels.shape[-1]

        elif self.label_class == 'R_Peak_Amplitude':
            self.dataset, self.labels, self.df = utils.get_regression_data(task=self.label_class)
            self.num_samples = len(self.dataset)
            self.num_classes = self.labels.shape[-1]

        else:
            self.df = pd.read_csv(self.data_dir/'ptbxl_database_enriched.csv',index_col=0)
            self.dataset = np.load(self.data_dir/'raw100.pkl',
                                allow_pickle=True)
            self.labels = np.load(
                self.data_dir/('multihot_'+self.label_class+'.npy'), allow_pickle=True)
        
        leads = [l.lower() for l in utils.leads]
        lead_ids = [leads.index(l.lower()) for l in self.lead_order]
      
        self.dataset = self.dataset[:, :, lead_ids]
        self.num_samples = len(self.dataset)
        self.num_classes = self.labels.shape[1]
             
    def prepare_data(self):
        pass

    def train_dataloader(self):
        train_dataset = SimpleDataset(self.dataset[(self.df.strat_fold.apply(
            lambda x: x in self.train_folds))], self.labels[(self.df.strat_fold.apply(lambda x: x in self.train_folds))], transform=RandomCrop(self.data_input_size))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, pin_memory=True, shuffle=self.shuffle_train, drop_last=self.drop_last)

        return train_loader

    def val_dataloader(self):
        val_dataset = SimpleDataset(
            self.dataset[self.df.strat_fold == self.val_fold], self.labels[self.df.strat_fold == self.val_fold], transform=CenterCrop(self.data_input_size))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=True)

        return val_loader

    def test_dataloader(self):
        test_dataset = SimpleDataset(
            self.dataset[self.df.strat_fold == self.test_fold], self.labels[self.df.strat_fold == self.test_fold], transform=CenterCrop(self.data_input_size))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 num_workers=self.num_workers, pin_memory=True)

        return test_loader

    def default_transforms(self):
        pass

def get_experiment_name(modelname, task):
    experiment_name = modelname + '_' + task

    return experiment_name

class RandomCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        data, label = sample

        timesteps = data.shape[1]
        assert(timesteps >= self.output_size)
        if(timesteps == self.output_size):
            start = 0
        else:
            start = random.randint(0, timesteps - self.output_size-1)

        data = data[:, start: start + self.output_size]
        return data, label

class CenterCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        data, label = sample

        timesteps = data.shape[1]
        assert(timesteps >= self.output_size)
        if(timesteps == self.output_size):
            start = 0
        else:
            start = (timesteps-self.output_size)//2

        data = data[:, start: start + self.output_size]
        return data, label

class SimpleDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            return self.transform((self.data[index].T, self.labels[index]))
        return self.data[index].T, self.labels[index]

    def __len__(self):
        return len(self.data)

@click.command()
@click.option('--data_dir', default='data/ptbxl/', help='path to dataset')
@click.option('--addon_dir', default='ptbxl_addons/', help='path to PTB-XL addons')
@click.option('--batch_size', default=128, help='batch_size')
@click.option('--modelname', default='lenet', help='model name')
@click.option('--task', default='subdiagnostic', help='task')
@click.option('--logdir', default='./output/logs', help='task')
@click.option('--epochs', default=100, type=int)
@click.option('--gpu', default=False, is_flag=True)
@click.option('--test_only', default=False, is_flag=True)
@click.option('--input_size', default=250)
@click.option('--checkpoint_path')
@click.option('--finetuning', default=False, is_flag=True)
@click.option('--lead_order')
def train_model(data_dir, addon_dir, batch_size, modelname, task, logdir, epochs, gpu, test_only, input_size, checkpoint_path, finetuning, lead_order):

    experiment_name = get_experiment_name(modelname, task)
    lead_order = lead_order.split(" ") if lead_order is not None else None

    regression_tasks = ['T_Wave_Amplitude', 'P_Wave_Amplitude', 'R_Peak_Amplitude']
    
    # data
    datamodule = ECGDataModule(
        batch_size,
        data_dir, 
        addon_dir,
        label_class=task,
        data_input_size=input_size,
        lead_order=lead_order,
    )

    # pytorch lightning module
    if task in regression_tasks:
        pl_model = get_model(modelname, datamodule.num_classes, loss_fn=F.mse_loss)
    else:
        pl_model = get_model(modelname, datamodule.num_classes)

    tb_logger = TensorBoardLogger(
        logdir, name=experiment_name, version="",)

    if task in regression_tasks:
        trainer = Trainer(
            logger=tb_logger,
            max_epochs=epochs,
            gpus=1 if gpu else 0,
            callbacks=[ModelCheckpoint(monitor='val/total_loss', mode='min', filename='best_model')],
        )
    else:
        trainer = Trainer(
            logger=tb_logger,
            max_epochs=epochs,
            gpus=1 if gpu else 0,
            callbacks=[ModelCheckpoint(monitor='val/val_macro', mode='max', filename='best_model')],
        )

    if checkpoint_path is not None:
        pl_model.load_from_checkpoint(
            checkpoint_path, modelname=modelname, lead_order=lead_order)

    pl_model.finetuning = finetuning
    
     # start training
    if not test_only:
        trainer.fit(pl_model, datamodule)
        trainer.save_checkpoint(os.path.join(
            logdir, experiment_name, "checkpoints", "model_lastepoch.ckpt"))

    _ = trainer.validate(pl_model, datamodule=datamodule)
    _ = trainer.test(pl_model, datamodule=datamodule)

if __name__ == '__main__':
    train_model()


