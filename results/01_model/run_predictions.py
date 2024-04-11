# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import argparse
from collections import defaultdict
import itertools
import logging
from pathlib import Path
import statistics
import sys
from typing import Iterable, Generator
import urllib.parse

import a2ze.data.lightning
import a2ze.models.lightning
import lightning.pytorch as ptl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import mlflow
import numpy as np
import omegaconf
import pandas as pd
import torch
import torchmetrics
from tqdm import tqdm

# %%
IS_NOTEBOOK = ('get_ipython' in locals()) and (get_ipython().__class__.__name__ == 'ZMQInteractiveShell')

if IS_NOTEBOOK:
    args = argparse.Namespace(
        tracking_uri = "http://127.0.0.1:6000",
        experiment_name = "full",
        dataset_path = "tmp/dataset",
        output_path = Path("tmp/predictions.notebook.tsv"),
        run_index = 0,
        num_workers = 6
    )
else:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "tracking_uri",
        help = "MLFlow tracking URI"
    )
    parser.add_argument(
        "experiment_name",
        help = "MLFlow experiment to grab runs from"
    )
    parser.add_argument(
        "dataset_path",
        help = "Path to dataset"
    )
    parser.add_argument(
        "output_path",
        help = "Path to write prediction TSV to",
        type = Path
    )
    parser.add_argument(
        "run_index",
        help = "Integer index of run in the experiment (sorted by ID) to run predictions on",
        type = int
    )

    parser.add_argument(
        '--num_workers',
        type = int,
        help = 'Number of processes to use for data loading',
        default = 6
    )

    args = parser.parse_args()


# %%
def iterate_search_pages(client: mlflow.client.MlflowClient, experiments: Iterable[mlflow.entities.Experiment]) -> Generator[mlflow.store.entities.PagedList[mlflow.entities.Run], None, None]:
    experiment_ids = [experiment.experiment_id for experiment in experiments]
    page = client.search_runs(experiment_ids = experiment_ids)

    while page:
        yield page
        page = False if page.token is None else client.search_runs(experiment_ids = experiment_ids, page_token = page.token)


# %%
client = mlflow.client.MlflowClient(tracking_uri = args.tracking_uri)

experiment = client.get_experiment_by_name(args.experiment_name)
if experiment is None:
    logging.error("Experiment %s not found", args.experiment_name)
    sys.exit(1)
experiments = [experiment]

# %%
runs = defaultdict(list)

for page in iterate_search_pages(client, experiments):
    for run in page:
        runs['id'].append(run.info.run_id)
        runs['end_time'].append(run.info.end_time)
        runs['architecture'].append(run.data.params['model/architecture'])
        runs['task'].append(run.data.params['data/task/name'])
        runs['validation_split'].append(run.data.params['training/validation_split_name'])
        # for metric in run.data.metrics:
        #     runs['history'][metric] = client.get_metric_history(
        #         run_id = run.info.run_id,
        #         key = metric
        #     )

runs = pd.DataFrame(runs).set_index('id', drop = True).sort_index(kind = 'stable')

# %%
predictions = []

run = next(runs.iloc[[args.run_index]].itertuples())

print("Running predictions for %s", str(run))

checkpoint_dirs = client.list_artifacts(run_id = run.Index, path = 'model/checkpoints')
# Lightning should be keeping only the best model checkpoint
if len(checkpoint_dirs) != 1:
    raise RuntimeError(f"Run {run.Index} has more than one checkpoint: {checkpoint_dirs}")
checkpoint_dir = Path(checkpoint_dirs[0].path)
checkpoint_relative_path = checkpoint_dir / f"{checkpoint_dir.name}.ckpt"
checkpoint_path = client.download_artifacts(run_id = run.Index, path = checkpoint_relative_path)
config = omegaconf.OmegaConf.load(client.download_artifacts(run_id = run.Index, path = 'config.yaml'))

model = getattr(a2ze.models.lightning, run.architecture).load_from_checkpoint(checkpoint_path, config = config)
# tqdm.auto only works with num_workers = 0
# pytorch claims it's tqdm (pytorch/pytorch#53703)
# tqdm claims it's PyTorch (tqdm/tqdm#1312)
# ¯\_(ツ)_/¯
datamodule = a2ze.data.lightning.Seq2ExpDataModule(
    root = args.dataset_path,
    task = config.data.task.name,
    validation_split_name = config.training.validation_split_name,
    batch_size = 256,
    upstream_context_bp = config.data.preprocessing.upstream_context_bp,
    input_type = config.data.preprocessing.input_type,
    task_type = config.data.task.type,
    num_workers = args.num_workers,
    keep_prop = 0.01 if IS_NOTEBOOK else 1.0
)

datamodule.setup('predict')
model.eval()
with torch.no_grad():
    predictions_run = []
    # predict the training data
    for batch in tqdm(datamodule.train_dataloader(), desc = "train"):
        predictions_batch = model.forward(batch)
        batch['target'] = batch['targets'].flatten()
        batch_df = pd.DataFrame(
            data = batch,
            columns = ['genome', 'id', 'orthogroup', 'target']
        )
        batch_df['prediction'] = predictions_batch.cpu()
        batch_df['split'] = 'train'
        predictions_run.append(batch_df)
    # predict the validation data
    for batch in tqdm(datamodule.val_dataloader(), desc = "val"):
        predictions_batch = model.forward(batch)
        batch['target'] = batch['targets'].flatten()
        batch_df = pd.DataFrame(
            data = batch,
            columns = ['genome', 'id', 'orthogroup', 'target']
        )
        batch_df['prediction'] = predictions_batch.cpu()
        batch_df['split'] = 'validate'
        predictions_run.append(batch_df)
    # predict the test data
    if not IS_NOTEBOOK:
        for batch in tqdm(datamodule.test_dataloader(), desc = "test"):
            predictions_batch = model.forward(batch)
            batch['target'] = batch['targets'].flatten()
            batch_df = pd.DataFrame(
                data = batch,
                columns = ['genome', 'id', 'orthogroup', 'target']
            )
            batch_df['prediction'] = predictions_batch.cpu()
            batch_df['split'] = 'test'
            predictions_run.append(batch_df)
del model
datamodule.teardown('predict')

predictions_run = pd.concat(predictions_run)
predictions_run['task'] = run.task
predictions_run['architecture'] = run.architecture
predictions_run['run_id'] = run.Index

# %%
predictions_run.to_csv(args.output_path, sep = "\t", index = False)

# %%
