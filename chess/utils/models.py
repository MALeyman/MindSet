

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNetMultiInput(nn.Module):
    def __init__(self):
        super().__init__()
        # Свёрточные слои для обработки позиции доски
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Полносвязный слой после свёрток
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        
        # Обработка castling прав — полносвязный слой
        self.castling_fc = nn.Linear(4, 32)
        
        # Общий полносвязный слой после объединения позиций и прав
        self.fc2 = nn.Linear(512 + 32, 1024)
        self.dropout = nn.Dropout(0.3)
        
        # Выходной слой — количество классов ходов, например 4096
        self.fc_out = nn.Linear(1024, 4096)

    def forward(self, board_input, castling_input):
        # board_input: [batch, 13, 8, 8]
        x = F.relu(self.bn1(self.conv1(board_input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # разворачиваем
        
        x = F.relu(self.fc1(x))  # [batch, 512]
        
        castling_features = F.relu(self.castling_fc(castling_input))  # [batch, 32]
        
        combined = torch.cat([x, castling_features], dim=1)  # [batch, 544]
        combined = F.relu(self.fc2(combined))
        combined = self.dropout(combined)
        
        out = self.fc_out(combined)  # [batch, 4096]
        return out



import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessNetImproved(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.resblock1 = ResidualBlock(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.resblock2 = ResidualBlock(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.resblock3 = ResidualBlock(256)

        self.fc1 = nn.Linear(256 * 8 * 8, 512)

        self.castling_fc = nn.Linear(4, 32)

        self.fc2 = nn.Linear(512 + 32, 1024)
        self.dropout = nn.Dropout(0.3)

        self.fc_out = nn.Linear(1024, 4096)

    def forward(self, board_input, castling_input):
        x = F.relu(self.bn1(self.conv1(board_input)))
        x = self.resblock1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.resblock2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.resblock3(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        castling_features = F.relu(self.castling_fc(castling_input))

        combined = torch.cat([x, castling_features], dim=1)
        combined = F.relu(self.fc2(combined))
        combined = self.dropout(combined)

        out = self.fc_out(combined)
        return out





import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=13, patch_size=1, embed_dim=256, board_size=8):
        super().__init__()
        assert board_size % patch_size == 0, "board size must be divisible by patch size"
        self.patch_size = patch_size
        self.num_patches = (board_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x2 = x2.transpose(0, 1)  # (N, B, E)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out.transpose(0, 1)  # skip connection
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class ChessVisionTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_dim=512, num_layers=6, patch_size=1, board_size=8, num_classes=4096):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=13, patch_size=patch_size, embed_dim=embed_dim, board_size=board_size)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)
        ])

        self.castling_fc = nn.Linear(4, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.castling_fc.weight)
        nn.init.zeros_(self.castling_fc.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, board_input, castling_input):
        B = board_input.shape[0]
        x = self.patch_embed(board_input)  # (B, N, E)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, E)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, E)
        x = x + self.pos_embed
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        castling_embed = self.castling_fc(castling_input)  # (B, E)
        x_cls = x[:, 0, :] + castling_embed
        x_norm = self.norm(x_cls)
        out = self.head(x_norm)  # (B, num_classes)
        return out

