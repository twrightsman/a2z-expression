import logging
from typing import Any, Callable, Optional

import numpy as np

from . import Dataset


class Flank():
    def __init__(
        self,
        dataset: Dataset,
        upstream: Optional[int] = None,
        downstream: Optional[int] = None
    ):
        if sum((a is None for a in (upstream, downstream))) != 1:
            raise ValueError("Either upstream or downstream must be specified, but not both")

        self._dataset = dataset

        if upstream is not None:
            self._func = self._upstream
        elif downstream is not None:
            self._func = self._downstream
        else:
            raise ValueError("This should not happen")

        self._length = upstream if upstream is not None else downstream

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self._func(obs)

    def _upstream(self, obs: dict[str, Any]) -> dict[str, Any]:
        max_end = self._dataset.get_genome(obs['genome']).get_reference_length(obs['seqid'])

        if obs['strand'] == '-':
            obs['start'] = obs['end']
            obs['end'] = min(obs['end'] + self._length, max_end)
        else:
            obs['end'] = obs['start']
            obs['start'] = max(obs['end'] - self._length, 0)

        return obs

    def _downstream(self, obs: dict[str, Any]) -> dict[str, Any]:
        max_end = self._dataset.get_genome(obs['genome']).get_reference_length(obs['seqid'])

        if obs['strand'] == '-':
            obs['end'] = obs['start']
            obs['start'] = max(obs['end'] - self._length, 0)
        else:
            obs['start'] = obs['end']
            obs['end'] = min(obs['end'] + self._length, max_end)

        return obs


class Extend():
    def __init__(
        self,
        dataset: Dataset,
        upstream: int = 0,
        downstream: int = 0
    ):
        self._dataset = dataset
        self._upstream = upstream
        self._downstream = downstream

        if (upstream == 0) and (downstream == 0):
            logging.warning("Created an Extend transform that does nothing (upstream/downstream both zero)")

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        max_end = self._dataset.get_genome(obs['genome']).get_reference_length(obs['seqid'])

        if obs['strand'] == '-':
            obs['start'] = max(obs['start'] - self._downstream, 0)
            obs['end'] = min(obs['end'] + self._upstream, max_end)
        else:
            obs['start'] = max(obs['start'] - self._upstream, 0)
            obs['end'] = min(obs['end'] + self._downstream, max_end)

        return obs


complement_DNA = str.maketrans('ACGT', 'TGCA')


class ExtractSequence():
    def __init__(
        self,
        dataset: Dataset,
        stranded: bool = True
    ):
        self._dataset = dataset
        self._stranded = stranded

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        genome = self._dataset.get_genome(obs['genome'])

        sequence = genome.fetch(reference = obs['seqid'], start = obs['start'], end = obs['end'])

        if self._stranded and ('strand' in obs) and (obs['strand'] == '-'):
            sequence = sequence.translate(complement_DNA)[::-1]

        obs['sequence'] = sequence

        return obs


class OneHotEncode():
    _vocab = {
        'A': np.array([1, 0, 0, 0]),
        'C': np.array([0, 1, 0, 0]),
        'G': np.array([0, 0, 1, 0]),
        'T': np.array([0, 0, 0, 1]),
        'W': np.array([0.5, 0, 0, 0.5]),
        'S': np.array([0, 0.5, 0.5, 0]),
        'M': np.array([0.5, 0.5, 0, 0]),
        'K': np.array([0, 0, 0.5, 0.5]),
        'R': np.array([0.5, 0, 0.5, 0]),
        'Y': np.array([0, 0.5, 0, 0.5]),
        'B': np.array([0, 1.0 / 3, 1.0 / 3, 1.0 / 3]),
        'D': np.array([1.0 / 3, 0, 1.0 / 3, 1.0 / 3]),
        'H': np.array([1.0 / 3, 1.0 / 3, 0, 1.0 / 3]),
        'V': np.array([1.0 / 3, 1.0 / 3, 1.0 / 3, 0]),
        'N': np.array([0.25, 0.25, 0.25, 0.25]),
    }

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        sequence = np.empty(shape = (len(obs['sequence']), 4))

        for i, base in enumerate(obs['sequence'].upper()):
            sequence[i, :] = self._vocab[base]

        obs['encoded'] = sequence

        return obs


class Pad():
    """
    Right pad a one-hot encoded DNA sequence matrix to the specified length.
    """

    def __init__(self, target_length: int):
        self._target_length = target_length

    @property
    def target_length(self) -> int:
        return self._target_length

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        padding_needed = self.target_length - obs['encoded'].shape[0]

        if padding_needed > 0:
            obs['encoded'] = np.pad(obs['encoded'], pad_width = ((0, padding_needed), (0, 0)), constant_values = 0.)

        return obs


class Trim():
    def __init__(self, max_len: int):
        self._max_len = max_len

    @property
    def max_len(self) -> int:
        return self._max_len

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs['encoded'] = obs['encoded'][:self.max_len, :]

        return obs


class Compose():
    """
    Simple implementation similar to torchvision's transforms.Compose
    """

    def __init__(self, transforms: list[Callable]):
        self._transforms = transforms

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        for transform in self._transforms:
            obs = transform(obs)

        return obs
