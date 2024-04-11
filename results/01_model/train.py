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
import logging
import secrets
import sys

import a2ze.models.lightning
from a2ze.data.lightning import Seq2ExpDataModule
from a2ze.utils import nested_to_flat, hash_dict_with_str_keys
import hydra
import lightning.pytorch as ptl
import omegaconf
import torch
import torchinfo

# %%
logger = logging.getLogger(__name__)

# %%
IS_NOTEBOOK = (
    'get_ipython' in locals()
) and (
    get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
)


# %%
def train(config: omegaconf.DictConfig):
    if omegaconf.OmegaConf.is_missing(config.training, "seed"):
        config.training.seed = secrets.randbits(32)

    ptl.seed_everything(config.training.seed, workers = True)

    datamodule = Seq2ExpDataModule(
        root = config.data.path,
        task = config.data.task.name,
        validation_split_name = config.training.validation_split_name,
        batch_size = config.training.batch_size,
        upstream_context_bp = config.data.preprocessing.upstream_context_bp,
        input_type = config.data.preprocessing.input_type,
        task_type = config.data.task.type,
        num_workers = config.data.preprocessing.num_workers,
        keep_genomes = config.training.keep_genomes,
        keep_prop = config.training.keep_prop
    )

    config_hash = hash_dict_with_str_keys(omegaconf.OmegaConf.to_container(config))

    model = getattr(a2ze.models.lightning, config.model.architecture)(config)

    if config.training.compile_model:
        if config.training.detect_anomaly:
            raise RuntimeError("Compiled models currently don't support detecting anomalies (pytorch/pytorch#100854)")
        model = torch.compile(model)

    if config.training.mlflow_tracking_uri is None:
        mlflow_tracking_uri = f"file://{hydra.utils.to_absolute_path(config.training.mlflow_save_dir)}"
    else:
        mlflow_tracking_uri = config.training.mlflow_tracking_uri

    logger_model = ptl.loggers.MLFlowLogger(
        experiment_name = config.training.experiment_name,
        run_name = f"{config.model.architecture}_{config.training.validation_split_name}_{config_hash}",
        tracking_uri = mlflow_tracking_uri,
        log_model = True
    )

    summary_model = torchinfo.summary(model)
    hyperparameters = {
        'model/trainable_parameters': summary_model.trainable_params
    }
    hyperparameters.update(nested_to_flat(config))
    logger_model.log_hyperparams(
        params = hyperparameters
    )

    # dump the run config to a YAML file artifact
    logger_model.experiment.log_dict(
        run_id = logger_model.run_id,
        dictionary = omegaconf.OmegaConf.to_container(config),
        artifact_file = "config.yaml"
    )

    trainer = ptl.Trainer(
        max_epochs = config.training.max_epochs,
        accelerator = config.training.accelerator,
        devices = config.training.num_devices,
        limit_train_batches = config.training.limit_train_batches,
        limit_val_batches = config.training.limit_val_batches,
        callbacks = [
            ptl.callbacks.EarlyStopping(
                monitor = "loss/validation",
                mode = 'min'
            ),
            ptl.callbacks.ModelCheckpoint(
                monitor = "loss/validation",
                save_top_k = 1
            )
        ],
        logger = logger_model,
        detect_anomaly = config.training.detect_anomaly,
        enable_model_summary = False,
        deterministic = True,
        default_root_dir = "tmp/lightning"
    )

    trainer.fit(
        model = model,
        datamodule = datamodule
    )


# %%
if IS_NOTEBOOK:
    with hydra.initialize(version_base = None, config_path = "config/"):
        cfg = hydra.compose(
            config_name = "config",
            overrides = [
                'data/task=exp-any',
                'model=FNetCompression',
                'training.experiment_name=notebook',
                'training.max_epochs=2',
                #'training.limit_train_batches=50',
                'training.limit_val_batches=10',
                'training.detect_anomaly=true',
                'training.mlflow_tracking_uri=http://127.0.0.1:6000',
                'training.keep_genomes=[Tripsacum_dactyloides/FL_9056069_6]',
                'training.keep_prop=0.1'
            ]
        )

        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)

            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                fmt = '{asctime} [{module}:{levelname}] {message}',
                style = '{'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        train(cfg)
elif __name__ == "__main__":
    hydra.main(version_base = None, config_path = "config/", config_name = "config")(train)()

# %%
