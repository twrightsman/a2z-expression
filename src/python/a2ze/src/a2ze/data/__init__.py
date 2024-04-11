import os
from pathlib import Path
from typing import Any, Callable, Sequence, List, Optional, Dict, Union

import numpy as np
import pandas as pd
import pysam
import torch.utils.data
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, task: str, split: str, keep_genomes: Optional[list[str]] = None, keep_prop: float = 1.0, random_state = None):
        path = Path(path)

        self._path = path
        self._task = task
        self._split = split

        data = pd.read_table(
            path / 'tasks' / task / f"{split}.tsv"
        )
        if keep_genomes is not None:
            keep_genomes = set(keep_genomes)
            self._data = data.loc[data['genome'].isin(keep_genomes)].copy()
        else:
            self._data = data

        if (keep_prop >= 0.0) and (keep_prop < 1.0):
            self._data = self._data.sample(frac = keep_prop, random_state = random_state)

        self._genomes = None

    def _load_genomes(self):
        genomes_dir = self._path / 'genomes'
        compressed_genomes = {os.sep.join(assembly_path.parts[len(genomes_dir.parts) : -1]): pysam.FastaFile(assembly_path) for assembly_path in genomes_dir.glob('**/assembly.fa.gz')}
        uncompressed_genomes = {os.sep.join(assembly_path.parts[len(genomes_dir.parts) : -1]): pysam.FastaFile(assembly_path) for assembly_path in genomes_dir.glob('**/assembly.fa')}
        # prioritize uncompressed genome references, if both present
        self._genomes = compressed_genomes | uncompressed_genomes

    @property
    def path(self) -> Path:
        return self._path

    @property
    def task(self) -> str:
        return self._task

    @property
    def split(self) -> str:
        return self._split

    def get_genome(self, genome: str) -> pysam.FastaFile:
        return self._genomes[genome]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._genomes is None:
            self._load_genomes()

        obs = dict(self._data.iloc[idx])
        obs['targets'] = np.array(obs['targets'])[np.newaxis]
        return obs

    def __len__(self) -> int:
        return len(self._data)


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, transform: Callable):
        super().__init__()

        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, key):
        return self._transform(self._dataset[key])


class CharacterTokenizer(PreTrainedTokenizer):
    """
    The CharacterTokenizer class was modified from HyenaDNA's upstream repository at
    https://github.com/HazyResearch/hyena-dna

    The manuscript describing the model is at
    https://arxiv.org/abs/2306.15794

    The code is licensed under the Apache-2.0 license.
    The full text for the Apache-2.0 license can be found at
    https://www.apache.org/licenses/LICENSE-2.0.html
    or from the Wayback Machine at
    https://web.archive.org/web/*/https://www.apache.org/licenses/LICENSE-2.0.html
    """
    def __init__(
        self,
        characters: Sequence[str],
        model_max_length: int,
        padding_side: str = "left",
        **kwargs,
    ):
        """Character tokenizer for Hugging Face transformers.
        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            model_max_length (int): Model maximum sequence length.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        super().__init__(
            bos_token=bos_token,
            eos_token=sep_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)
