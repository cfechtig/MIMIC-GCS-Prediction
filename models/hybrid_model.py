import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridModel(nn.Module):
    def __init__(
        self,
        static_dim,
        seq_feature_dim,
        task="regression",
        lstm_hidden=64,
        static_hidden=[64, 32],
        dropout=0.2,
        num_classes=1
    ):
        super().__init__()
        self.task = task.lower()
        assert self.task in ["regression", "classification", "multiclass"], "Task must be \"regression\", \"classification\", or \"multiclass\""

        self.num_classes = num_classes

        # Sequential branch
        self.lstm = nn.LSTM(input_size=seq_feature_dim, hidden_size=lstm_hidden, batch_first=True)
        self.seq_fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Static branch
        static_layers = []
        input_dim = static_dim
        for hidden_dim in static_hidden:
            static_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        self.static_fc = nn.Sequential(*static_layers)

        # Combined head
        combined_dim = 32 + static_hidden[-1]
        self.final = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes if task == "multiclass" else 1)
        )

    def forward(self, x_static, x_seq):
        _, (h_n, _) = self.lstm(x_seq)
        x_seq_out = self.seq_fc(h_n[-1])
        x_static_out = self.static_fc(x_static)
        x = torch.cat([x_seq_out, x_static_out], dim=1)
        x = self.final(x)

        if self.task == "regression":
            return x.squeeze(1)
        elif self.task == "classification":
            return torch.sigmoid(x).squeeze(1)
        elif self.task == "multiclass":
            return F.log_softmax(x, dim=1)
