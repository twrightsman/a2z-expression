"""
FNetCompression

Code roughly translated from Keras implementation:
https://github.com/vittoriopipoli/FNetCompression

Paper: https://gattanasio.cc/publication/2023-squeeze-and-learn/2023-squeeze-and-learn.pdf
"""

from collections import OrderedDict
import math
from typing import Optional

import torch


class FNetBlock(torch.nn.Module):
    def __init__(
        self,
        max_length: int,
        embedding_dim: int,
        feedforward_dim: int,
        low_pass_cutoff: Optional[int] = None
    ):
        super().__init__()

        self._max_length = max_length
        self._embedding_dim = embedding_dim
        self._feedforward_dim = feedforward_dim
        self._low_pass_cutoff = low_pass_cutoff

        self.ffn = torch.nn.Sequential(OrderedDict((
            ('ffn1', torch.nn.Linear(
                in_features = self._embedding_dim,
                out_features = self._feedforward_dim
            )),
            ('gelu', torch.nn.GELU()),
            ('ffn2', torch.nn.Linear(
                in_features = self._feedforward_dim,
                out_features = self._embedding_dim
            ))
        )))

        self.layernorm1 = torch.nn.LayerNorm(
            normalized_shape = torch.Size((self._embedding_dim, self._max_length)),
            eps = 1e-6
        )
        self.layernorm2 = torch.nn.LayerNorm(
            normalized_shape = torch.Size((self._embedding_dim, self._max_length)),
            eps = 1e-6
        )

    def forward(self, x: torch.Tensor):
        x_fft = torch.real(torch.fft.fft2(x))
        x = self.layernorm1(x + x_fft)

        if self._low_pass_cutoff is not None:
            cut = self._low_pass_cutoff
            # grab an extra on the left if odd length
            out_left = x[:, :, :cut + (self._max_length % 2)]
            out_right = x[:, :, -cut:]
            x = torch.concatenate([
                out_left,
                out_right
            ], axis = -1)

        x_ffn = self.ffn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.layernorm2(x + x_ffn)

        return x


class FNetCompression(torch.nn.Module):
    def __init__(
        self,
        max_length: int,
        vocab_size: Optional[int] = None,
        input_type: str = "one-hot",
        embed_dim: int = 32,
        num_heads: int = 4,
        pool_size: int = 30,
        t_rate: float = 0.1,
        num_fnet_layers: int = 3,
        feedforward_dim: int = 64,
        low_pass_filter: bool = True,
        num_transformer_layers: int = 2,
        final_fc_dim: int = 64
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._max_length = max_length
        self._input_type = input_type
        self._embedding_dim = embed_dim
        self._num_heads = num_heads
        self._pool_size = pool_size
        self._t_rate = t_rate
        self._num_fnet_layers = num_fnet_layers
        self._feedforward_dim = feedforward_dim
        self._low_pass_cutoff = math.ceil(max_length / pool_size) // 2 if low_pass_filter else None
        self._num_transformer_layers = num_transformer_layers
        self._final_fc_dim = final_fc_dim
        
        # embedding
        if self._input_type == "one-hot":
            self.embedding = torch.nn.Conv1d(
                in_channels = 4,
                out_channels = self._embedding_dim,
                kernel_size = 1
            )
        elif self._input_type == "tokens":
            if self._vocab_size is None:
                raise ValueError("vocab_size must be provided if input_type is 'tokens'")
            self.embedding = torch.nn.Embedding(
                num_embeddings = self._vocab_size,
                embedding_dim = self._embedding_dim
            )
        else:
            raise ValueError("input_type must be either 'tokens' or 'one-hot'")

        # convolutions
        self.conv1_1 = torch.nn.Conv1d(
            in_channels = self._embedding_dim,
            out_channels = self._embedding_dim,
            kernel_size = 6,
            padding = 'same',
            groups = self._num_heads
        )
        self.conv1_2 = torch.nn.Conv1d(
            in_channels = self._embedding_dim,
            out_channels = self._embedding_dim,
            kernel_size = 9,
            padding = 'same',
            groups = self._num_heads
        )
        
        self.fc1 = torch.nn.Linear(
            in_features = self._embedding_dim * 2,
            out_features = self._embedding_dim
        )
        # https://stackoverflow.com/a/75141540
        self.bn1 = torch.nn.BatchNorm1d(
            num_features = None,
            affine = False,
            track_running_stats = False
        )
        
        self.embedding_position = torch.nn.Embedding(
            num_embeddings = math.ceil(self._max_length / self._pool_size),
            embedding_dim = self._embedding_dim
        )
        self.dropout = torch.nn.Dropout(p = self._t_rate)

        self.fnet = torch.nn.Sequential(OrderedDict(
            ((f"fnet{i}", FNetBlock(
                max_length = math.ceil(self._max_length / self._pool_size),
                embedding_dim = self._embedding_dim,
                feedforward_dim = self._feedforward_dim,
                low_pass_cutoff = self._low_pass_cutoff
            )) for i in range(self._num_fnet_layers))
        ))

        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model = self._embedding_dim,
                nhead = 4,
                activation = torch.nn.functional.gelu,
                dim_feedforward = self._feedforward_dim,
                dropout = self._t_rate,
                layer_norm_eps = 1e-6
            ),
            num_layers = self._num_transformer_layers
        )

        self.fc2 = torch.nn.Linear(
            in_features = self._embedding_dim,
            out_features = self._final_fc_dim
        )
        self.fc3 = torch.nn.Linear(
            in_features = self._final_fc_dim,
            out_features = self._final_fc_dim
        )
        self.fc4 = torch.nn.Linear(
            in_features = self._final_fc_dim,
            out_features = 1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        if self._input_type == "one-hot":
            x = torch.nn.functional.relu(x)
        else:
            # permute to channels first for PyTorch conv
            x = x.permute(0, 2, 1)

        x_conv1_1 = torch.nn.functional.relu(self.conv1_1(x))
        x_conv1_2 = torch.nn.functional.relu(self.conv1_2(x))
        x_conv1_12 = torch.concatenate([x_conv1_1, x_conv1_2], axis = 1).permute(0, 2, 1)
        x_conv1 = torch.nn.functional.relu(self.fc1(x_conv1_12))
        x = x_conv1.permute(0, 2, 1) + x

        x = torch.nn.functional.avg_pool1d(
            input = x,
            kernel_size = self._pool_size,
            ceil_mode = True
        )
        x = self.bn1(x)

        embedding_position = self.embedding_position(
            torch.arange(x.shape[2], device = x.device)
        ).permute(1, 0)
        x = self.dropout(x + embedding_position)

        x = self.fnet(x)

        # concatenate "CLS" token
        torch.concatenate([
            x,
            torch.full(
                size = (*x.shape[:2], 1),
                fill_value = 1,
                dtype = x.dtype,
                device = x.device
            )
        ], axis = -1)

        # transformer expects (seq x batch x emb)
        x = self.transformer(x.permute(2, 0, 1))
        # global average pooling across sequence
        x = x.mean(dim = 0)
        x = torch.nn.functional.tanh(x)

        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.nn.functional.gelu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x
