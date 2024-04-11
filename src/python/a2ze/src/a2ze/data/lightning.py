import inspect
import operator
from pathlib import Path
from typing import Any, Callable, Optional

import a2ze.data
import a2ze.data.transforms as transforms
import lightning.pytorch as ptl
import numpy as np
import torch
from transformers import PreTrainedTokenizer


class Tokenize:
    def __init__(self, tokenizer: PreTrainedTokenizer, **tokenizer_kwargs):
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        if obs['sequence'] or ('max_length' not in self._tokenizer_kwargs):
            obs['tokens'] = np.array(
                self._tokenizer(obs['sequence'], **self._tokenizer_kwargs)['input_ids'],
                dtype = np.int64
            )
        else:
            # tokenizer doesn't emit only padding with empty sequence
            # handle that here
            obs['tokens'] = np.full(
                shape = (self._tokenizer_kwargs['max_length'],),
                fill_value = self._tokenizer.pad_token_id,
                dtype = np.int64
            )
        return obs


class ToTensor:
    def __init__(self, key: str, **kwargs):
        self._key = key
        self._kwargs = kwargs

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs[self._key] = torch.tensor(obs[self._key], **self._kwargs)
        return obs


class Map:
    def __init__(self, *args, key: str, function: Callable, **kwargs):
        self._key = key
        self._function = function
        self._args = args
        self._kwargs = kwargs

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs[self._key] = self._function(obs[self._key], *self._args, **self._kwargs)
        return obs


class Permute:
    def __init__(self, dimensions: tuple[int]):
        self._dimensions = dimensions

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs['encoded'] = obs['encoded'].permute(*self._dimensions)
        return obs


class MapKey:
    def __init__(self, from_key: str, to_key: str):
        self._from_key = from_key
        self._to_key = to_key

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs[self._to_key] = obs[self._from_key]
        return obs


transforms.Tokenize = Tokenize
transforms.ToTensor = ToTensor
transforms.Map = Map
transforms.Permute = Permute
transforms.MapKey = MapKey


class Seq2ExpDataModule(ptl.LightningDataModule):
    def __init__(
        self,
        root: Path,
        task: str,
        validation_split_name: str,
        batch_size: int,
        upstream_context_bp: int,
        input_type: str,
        task_type: str,
        num_workers: int = 0,
        keep_genomes: Optional[list[str]] = None,
        keep_prop: float = 1.0
    ):
        super().__init__()
        self._root = root
        self._task = task
        self._batch_size = batch_size
        self._validation_split_name = validation_split_name
        self._keep_genomes = keep_genomes
        self._keep_prop = keep_prop

        if input_type == "1-hot":
            # 1-hot encoded input
            self._transforms = [
                ('Flank', (), {'upstream': upstream_context_bp}),
                ('ExtractSequence', (), {}),
                ('OneHotEncode', (), {}),
                ('Pad', (), {'target_length': upstream_context_bp}),
                ('Trim', (), {'max_len': upstream_context_bp}),
                ('ToTensor', (), {'key': 'encoded', 'dtype': torch.float32}),
                ('ToTensor', (), {'key': 'targets', 'dtype': torch.float32}),
                ('Permute', (), {'dimensions': (1, 0)}),
                ('MapKey', (), {'from_key': 'encoded', 'to_key': 'inputs'})
            ]
            self._tokenizer = None
        elif input_type == "tokens":
            # tokenized input
            self._tokenizer = a2ze.data.CharacterTokenizer(
                characters = ['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
                model_max_length = upstream_context_bp,  # to account for special tokens, like EOS
                add_special_tokens = False,  # we handle special tokens elsewhere
                padding_side = 'left', # since HyenaDNA is causal, we pad on the left
            )

            self._transforms = [
                ('Flank', (), {'upstream': upstream_context_bp}),
                ('ExtractSequence', (), {}),
                ('Tokenize', (), {
                    'tokenizer': self._tokenizer,
                    'add_special_tokens': False,
                    'padding': 'max_length',
                    'max_length': upstream_context_bp,
                    'truncation': True
                }),
                ('ToTensor', (), {'key': 'tokens', 'dtype': torch.int64}),
                ('ToTensor', (), {'key': 'targets', 'dtype': torch.float32}),
                ('MapKey', (), {'from_key': 'tokens', 'to_key': 'inputs'})
            ]

        if task_type == "regression":
            # log10(x + 1) transform regression outputs
            self._transforms += [
                ('Map', (1,), {'key': 'targets', 'function': operator.add}),
                ('Map', (), {'key': 'targets', 'function': torch.log10})
            ]
        
        self._num_workers = num_workers

    @property
    def tokenizer(self):
        return self._tokenizer
    
    def setup(self, stage: str):
        dataset_train = a2ze.data.Dataset(
            path = self._root,
            task = self._task,
            split = 'train',
            keep_genomes = self._keep_genomes,
            keep_prop = self._keep_prop
        )
        dataset_validate = a2ze.data.Dataset(
            path = self._root, task = self._task, split = self._validation_split_name
        )
        dataset_test = a2ze.data.Dataset(
            path = self._root, task = self._task, split = 'test'
        )

        self.dataset_train = self._get_transformed_dataset(dataset_train)
        self.dataset_validate = self._get_transformed_dataset(dataset_validate)
        self.dataset_test = self._get_transformed_dataset(dataset_test)

    def _get_transformed_dataset(self, dataset: a2ze.data.Dataset):
        to_compose = []
        for transform in self._transforms:
            transform_class_name, args, kwargs = transform
            transform_class = getattr(transforms, transform_class_name)
            if 'dataset' in inspect.getfullargspec(transform_class).args:
                to_compose.append(transform_class(*args, dataset = dataset, **kwargs))
            else:
                to_compose.append(transform_class(*args, **kwargs))
        return a2ze.data.TransformedDataset(
            dataset = dataset,
            transform = transforms.Compose(to_compose)
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset = self.dataset_train,
            batch_size = self._batch_size,
            num_workers = self._num_workers,
            shuffle = True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset = self.dataset_validate,
            batch_size = self._batch_size,
            num_workers = self._num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset = self.dataset_test,
            batch_size = self._batch_size,
            num_workers = self._num_workers
        )

    def teardown(self, stage: str):
        del self.dataset_train
        del self.dataset_validate
        del self.dataset_test
