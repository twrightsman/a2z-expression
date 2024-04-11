import torch

class DanQ(torch.nn.Module):
    def __init__(
        self,
        conv_channels: int = 320,
        conv_kernel_size: int = 26,
        pool_kernel_size: int = 13,
        pool_stride: int = 13,
        dropout1_rate: float = 0.2,
        lstm_hidden_size: int = 320,
        dropout2_rate: float = 0.5,
        classifier: bool = False,
        n_out_features: int = 1
    ):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(
            in_channels = 4,
            out_channels = conv_channels,
            kernel_size = conv_kernel_size
        )
        self.maxpool1 = torch.nn.MaxPool1d(
            kernel_size = pool_kernel_size,
            stride = pool_stride,
            ceil_mode = True
        )
        self.dropout1 = torch.nn.Dropout(
            p = dropout1_rate
        )
        self.lstm = torch.nn.LSTM(
            input_size = conv_channels,
            hidden_size = lstm_hidden_size,
            bidirectional = True,
            batch_first = True
        )
        self.dropout2 = torch.nn.Dropout(
            p = dropout2_rate
        )
        self.linear1 = torch.nn.Linear(
            in_features = conv_channels * 2,
            out_features = n_out_features
        )
        self.activation_final = torch.nn.functional.sigmoid if classifier else torch.nn.functional.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        output, (hn, cn) = self.lstm(x.transpose(-1, -2))
        x = hn.transpose(0, -2).flatten(-2, -1)
        x = self.dropout2(x)
        x = self.linear1(x)

        return self.activation_final(x)
