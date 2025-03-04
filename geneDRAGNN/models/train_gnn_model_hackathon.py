# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import json
import os
import datetime

from tqdm import tqdm

np.random.seed(314159) # set random seed

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from torch_geometric.data import Data
import torch_geometric.nn.conv as pygconv
import torch_geometric.loader

import wandb

import models_hackathon as models
from data_utils import read_data, create_data
import model_utils
import sys

# +
base_path = '../data/final_data/'

model_creator_dict = {'SGConv': models.create_SGConv_GNN,
                      'GraphSAGE': models.create_GraphSAGE_GNN,
                      'TAG': models.create_TAG_GNN,
                      'ClusterGCN': models.create_clusterGCN_GNN,
                      'MLP': models.create_MLP}
# -

node_dataset, edge_list, labels = read_data(node_filepath="../data/final_data/node_node2vec_data.csv",
                                            label_filepath="../data/final_data/training_labels_trials.csv",
                                            edgelist_path="../data/final_data/ls-fgin_edge_list.edg",
                                            feats_type='nodeonly')
edge_list = edge_list.astype('int32')

num_classes = 2
num_features = len(node_dataset.columns)

import os
notebook_name = 'train_gnn_model.ipynb'
os.environ['WANDB_NOTEBOOK_NAME'] = notebook_name

# +
import gc
from sklearn import metrics

def updateModel(model):
    layerTypes = [pygconv.GCNConv, pygconv.SAGEConv, pygconv.GATConv, pygconv.TransformerConv, pygconv.HGTConv]

    if(model_name == 'GraphSAGE'):
        model.model.convs[0] = layerTypes[0](105, 256)
        model.model.convs[1] = layerTypes[0](256, 256)
        model.model.convs[2] = layerTypes[0](256, 256)
    return model

def run_trials(create_model, model_name, start_trial=0, end_trial=100, n_epochs=500, log=False, log_project=None):

    if log:
        # dt_string = str(datetime.datetime.today()).replace(' ', '_')
        if log_project is None:
            print('Enter the name of the log project: ')
            log_project = input()



    # model info
    model = create_model()
    model = updateModel(model)
    model_summary = pl.utilities.model_summary.summarize(model, max_depth=4)
    model_summary_str = str(model_summary)
    num_trainable_params = model_summary.trainable_parameters
    


    print(model_summary_str)

    train_reports = []
    test_reports = []
    roc_data = []

    for trial in tqdm(range(start_trial, end_trial + 1)):

        print(f'running trial {str(trial)}')
        data = create_data(node_dataset, edge_list, labels, f'label_{trial}', test_size=0.2, val_size=0.1)


        model = create_model()
        print('starting a model of format:')
        print(model)

        model = updateModel(model)

        print('changed to format:')
        print(model)


        if log:
            n_zfills = int(np.ceil(np.log10(100)))
            log_name = f'{log_project}_trial{str(trial).zfill(n_zfills)}'

            logger = WandbLogger(name=log_name, project=log_project, log_model="\all\\", save_dir='wandb_projects')

            logger.log_metrics({'model_summary_str': model_summary_str,
                                'num_trainable_params': num_trainable_params})
            
            # log random train-val-test split
            logger.log_metrics({'train_mask': data.train_mask, 'val_mask': data.val_mask, 'test_mask': data.test_mask})

        else:
            logger = False

        AVAIL_GPUS = min(1, torch.cuda.device_count())

        data_loader = torch_geometric.loader.DataLoader([data], batch_size=1, num_workers=os.cpu_count())

        trainer = pl.Trainer(
                    callbacks=[ModelCheckpoint(save_weights_only=False, mode="max", monitor='val_acc')],
                    accelerator="gpu", devices=1,
                    max_epochs=n_epochs,
                    logger=logger,
                    enable_model_summary=False
                    # progress_bar_refresh_rate=0,
                    )
        trainer.fit(model, data_loader, data_loader)

        model = models.LitGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        train_report, test_report = model_utils.evaluate_model(model, data, logger=logger)

        train_reports.append(train_report)
        test_reports.append(test_report)

        model.to(device='cuda')
        logits, _, _ = model.forward(data.to(device='cuda'))

        preds = logits[data.test_mask][:, 1].cpu().detach().numpy()
        y = data.y[data.test_mask].cpu().detach().numpy()

        fpr, tpr, thresholds = metrics.roc_curve(y, preds)
        auc_score = metrics.roc_auc_score(y, preds)

        roc_data.append({'fpr': fpr, 'tpr': tpr, 'threholds': thresholds, 'auc': auc_score})

        if log:
            logger.log_metrics({'auc_test': auc_score, 'fpr_test': fpr, 
                                'tpr_test': tpr, 'roc_thres': thresholds})

        if log:
            wandb.save('modeling_gnn.ipynb')
            wandb.finish(quiet=True)

        del model, data_loader, trainer, data
        gc.collect()

        print('memory allocated: ', torch.cuda.memory_allocated())
        print('memory reserved: ', torch.cuda.memory_reserved())
        torch.cuda.empty_cache()
        print('\\nafter empty_cache:')
        print('memory allocated: ', torch.cuda.memory_allocated())
        print('memory reserved: ', torch.cuda.memory_reserved())



    return train_reports, test_reports, roc_data

# +
## TRAIN AND EVALUATE MODEL

model_name = 'GraphSAGE'

'''
model_creator_dict = {'SGConv': models.create_SGConv_GNN,
                      'GraphSAGE': models.create_GraphSAGE_GNN,
                      'TAG': models.create_TAG_GNN,
                      'ClusterGCN': models.create_clusterGCN_GNN,
                      'MLP': models.create_MLP}
'''

create_model = model_creator_dict[model_name]

trialDetailsString = sys.argv[1]

log_project_name = f'{model_name}' + trialDetailsString
# run multiple trials
train_reports, test_reports, roc_data = run_trials(lambda: create_model(model_name, num_features, num_classes), model_name, start_trial=0, end_trial=0,
                                         n_epochs=250, log=False, log_project=log_project_name)

# save reports from trials to json
model_utils.save_reports(f'project_reports/{log_project_name}_reports', train_reports, test_reports)
np.save(f'project_reports/{log_project_name}_roc', roc_data)
