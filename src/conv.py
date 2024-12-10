import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        # self.residual = (in_channels == out_channels)

    def forward(self, x):
        out = self.conv(x)
        # if self.residual:
        #     out = out + x
        return out


class ResBlock(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(size, size)

    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return residual + out


class QNet(nn.Module):
    def __init__(self, action_size, hidden_size=128):
        super().__init__()


        self.feature_extractor = nn.Sequential(
            ConvBlock(2, 32, kernel_size=3, padding=1),
            ConvBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64, kernel_size=3, padding=1),
        )


        self.position_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.local_feature_extractor = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )


        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(8, 8))
        self.conv_flattened_size = 64 * 8 * 8 

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.conv_flattened_size + 32 + 32, hidden_size),
            nn.ReLU()
        )

        layers = []
        num_res_blocks = 2
        for _ in range(num_res_blocks):
            layers.append(ResBlock(hidden_size))
        self.res_fc = nn.Sequential(*layers)

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, action_size)
        )

    def normalize_positions(self, positions, grid_size):
        return positions.float() / (grid_size - 1)  

    def forward(self, grid_status, gaussian_grid, drone_pos, local):
        batch_size = grid_status.size(0)
        grid_size = grid_status.size(-1)  

        normalized_pos = self.normalize_positions(drone_pos, grid_size)

        grid_input = torch.cat([grid_status.unsqueeze(1), gaussian_grid.unsqueeze(1)], dim=1)
      
        conv_features = self.feature_extractor(grid_input)
        conv_features = self.adaptive_avg_pool(conv_features)
        conv_flattened = conv_features.view(batch_size, -1)

        local_features = self.local_feature_extractor(local)

        pos_features = self.position_encoder(normalized_pos)

        combined_features = torch.cat([conv_flattened, local_features, pos_features], dim=1)
        fused_features = self.fusion_layer(combined_features)

        x = self.res_fc(fused_features)
      
        return self.action_head(x)

