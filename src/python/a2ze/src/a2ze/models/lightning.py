from collections import OrderedDict

import a2ze.models
import hydra
import lightning.pytorch as ptl
import omegaconf
import torch
import torchmetrics


class Seq2Exp(ptl.LightningModule):
    def __init__(self, config: omegaconf.DictConfig):
        super().__init__()

        for key in config.training.optimizer:
            self.hparams[f"training/optimizer/{key}"] = config.training.optimizer[key]

        if config.data.task.type == "classification":
            self.loss = torch.nn.functional.binary_cross_entropy
            self._on_validation_epoch_end = self._on_validation_epoch_end_classifier
        else:
            self.loss = torch.nn.functional.mse_loss

        # set up epoch output lists
        self._validation_step_outputs = {
            'predictions': [],
            'targets': []
        }

    def training_step(self, batch, batch_idx):
        y = batch['targets']
        y_hat = self.module(batch['inputs'])

        loss = self.loss(
            input = y_hat,
            target = y
        )
        self.log("loss/training", loss, batch_size = y.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['targets']
        y_hat = self.module(batch['inputs'])

        loss = self.loss(
            input = y_hat,
            target = y
        )
        self.log("loss/validation", loss, batch_size = y.shape[0])

        self._validation_step_outputs['targets'].append(y)
        self._validation_step_outputs['predictions'].append(y_hat)

        return loss

    def on_validation_epoch_end(self):
        self._on_validation_epoch_end()
        self._validation_step_outputs['targets'].clear()
        self._validation_step_outputs['predictions'].clear()

    def _on_validation_epoch_end_classifier(self):
        y = torch.concat(self._validation_step_outputs['targets']).int()
        y_hat = torch.concat(self._validation_step_outputs['predictions'])
        self.log(
            name = "accuracy",
            value = torchmetrics.functional.accuracy(
                preds = y_hat,
                target = y,
                task = 'binary'
            ),
            sync_dist = True
        )
        self.log(
            name = "AUROC",
            value = torchmetrics.functional.auroc(
                preds = y_hat,
                target = y,
                task = 'binary'
            ),
            sync_dist = True
        )
        self.log(
            name = "AUPR",
            value = torchmetrics.functional.average_precision(
                preds = y_hat,
                target = y,
                task = 'binary'
            ),
            sync_dist = True
        )

    def _on_validation_epoch_end(self):
        y = torch.concat(self._validation_step_outputs['targets'])
        y_hat = torch.concat(self._validation_step_outputs['predictions'])
        self.log(
            name = "correlation/pearson/validation",
            value = torchmetrics.functional.pearson_corrcoef(
                preds = y_hat,
                target = y
            ),
            sync_dist = True
        )
        self.log(
            name = "correlation/spearman/validation",
            value = torchmetrics.functional.spearman_corrcoef(
                preds = y_hat,
                target = y
            ),
            sync_dist = True
        )

    def forward(self, batch):
        return self.module(batch['inputs'].to(self.device))

    def configure_optimizers(self):
        kwargs = {key.split('/')[-1]: value for key, value in self.hparams.items() if key.startswith('training/optimizer')}
        del kwargs['name']

        optimizer = getattr(torch.optim, self.hparams['training/optimizer/name'])(
            params = self.parameters(),
            **kwargs
        )
        return optimizer


class DanQ(Seq2Exp):
    def __init__(self, config: omegaconf.DictConfig):
        super().__init__(config)

        self.module = a2ze.models.DanQ(
            **omegaconf.OmegaConf.to_container(config.model.hyperparameters),
            classifier = config.data.task.type == "classification"
        )


class HyenaDNA(Seq2Exp):
    def __init__(self, config: omegaconf.DictConfig):
        super().__init__(config)

        if config.data.task.type == "classification":
            self.module = torch.nn.Sequential(OrderedDict([
                ('hyena', a2ze.models.HyenaDNA(
                    **omegaconf.OmegaConf.to_container(config.model.hyperparameters),
                    use_head = True,
                    n_classes = 1
                )),
                ('sigmoid', torch.nn.Sigmoid())
            ]))
        else:
            self.module = torch.nn.Sequential(OrderedDict([
                ('hyena', a2ze.models.HyenaDNA(
                    **omegaconf.OmegaConf.to_container(config.model.hyperparameters),
                    use_head = True,
                    n_classes = 1
                ))
            ]))


class Miniformer(Seq2Exp):
    def __init__(self, config: omegaconf.DictConfig):
        super().__init__(config)

        if config.data.task.type == "classification":
            self.module = torch.nn.Sequential(OrderedDict([
                ('miniformer', a2ze.models.Miniformer(
                    **omegaconf.OmegaConf.to_container(config.model.hyperparameters)
                )),
                ('sigmoid', torch.nn.Sigmoid())
            ]))
        else:
            self.module = a2ze.models.Miniformer(
                **omegaconf.OmegaConf.to_container(config.model.hyperparameters)
            )


class FNetCompression(Seq2Exp):
    def __init__(self, config: omegaconf.DictConfig):
        super().__init__(config)

        tokenizer = hydra.utils.instantiate(config.data.preprocessing.tokenizer)
        if config.data.task.type == "classification":
            self.module = torch.nn.Sequential(OrderedDict([
                ('FNetCompression', a2ze.models.FNetCompression(
                    **omegaconf.OmegaConf.to_container(config.model.hyperparameters),
                    input_type = "tokens",
                    vocab_size = tokenizer.vocab_size
                )),
                ('sigmoid', torch.nn.Sigmoid())
            ]))
        else:
            self.module = torch.nn.Sequential(OrderedDict([
                ('FNetCompression', a2ze.models.FNetCompression(
                    **omegaconf.OmegaConf.to_container(config.model.hyperparameters),
                    input_type = "tokens",
                    vocab_size = tokenizer.vocab_size
                ))
            ]))
