import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data


# class ChessOneHotDataset(Dataset):
#     def __init__(self, csv_file):
#         self.data = pd.read_csv(csv_file)
#         # Конвертация pandas dataframe в numpy массивы для эффективности
#         self.features = self.data.drop(columns=['move_idx']).values.astype(np.float32)
#         self.labels = self.data['move_idx'].values.astype(np.int64)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         feature = torch.tensor(self.features[idx])
#         label = torch.tensor(self.labels[idx])
#         return feature, label


class ChessDataset(Dataset):
    """
    Класс датасета для загрузки шахматных данных из CSV-файла для обучения модели.

    Атрибуты:
    - data (pd.DataFrame): загруженный CSV с данными.
    - board_features (np.ndarray): признаки доски без колонок с рокировками и метками ходов, в формате float32.
    - castling_features (np.ndarray): признаки рокировки из соответствующих колонок, в формате float32.
    - labels (np.ndarray): метки ходов (индексы ходов), в формате int64.

    Методы:
    - __len__(): возвращает количество примеров в датасете.
    - __getitem__(idx): возвращает из датасета кортеж:
        (тензор признаков доски [13,8,8], тензор признаков рокировки [4], тензор метки хода).
    """
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.board_features = self.data.drop(columns=['castle_K', 'castle_Q', 'castle_k', 'castle_q', 'move_idx']).values.astype('float32')
        self.castling_features = self.data[['castle_K', 'castle_Q', 'castle_k', 'castle_q']].values.astype('float32')
        self.labels = self.data['move_idx'].values.astype('int64')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_feat = self.board_features[idx].reshape(13, 8, 8)
        castling_feat = self.castling_features[idx]
        label = self.labels[idx]
        return torch.tensor(board_feat), torch.tensor(castling_feat), torch.tensor(label)



def create_data_loaders_from_csv(dataset, batch_size=32, val_split=0.2):
    ''' 
    Создание лоадеров
    '''
    # dataset = ChessDataset(csv_file)

    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size)
    split = int(dataset_size * (1 - val_split))

    train_indices = indices[:split]
    val_indices = indices[split:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    return train_loader, val_loader


#  ==================  для графовой модели =======================

def create_chessboard_edges():
    edges = []
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        nidx = nr * 8 + nc
                        edges.append([idx, nidx])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


class ChessGNNDataset(Dataset):
    ''' 
    Датасет для графовой модели
    '''
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # Извлекаем оцифрованные особенности доски и рокировок
        # board_features shape: [num_samples, 13*8*8]
        self.board_features = self.data.drop(columns=['castle_K', 'castle_Q', 'castle_k', 'castle_q', 'move_idx']).values.astype(np.float32)
        self.castling_features = self.data[['castle_K', 'castle_Q', 'castle_k', 'castle_q']].values.astype(np.float32)
        self.labels = self.data['move_idx'].values.astype(np.int64)
        self.edge_index = create_chessboard_edges()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Получаем 13x8x8 для доски
        board_feat = self.board_features[idx].reshape(13, 8, 8)
        # Добавляем рокировки ко всем клеткам, расширяя до 4 каналов, дублируя по 64 клеткам
        castling_feat = self.castling_features[idx]
        castling_feat_expanded = np.tile(castling_feat, (64, 1))  # shape (64, 4)

        # Узлы: для каждой клетки объединяем фичи фигуры + рокировки: (64, 13+4=17)
        nodes_feat = np.concatenate([
            board_feat.reshape(13, 64).T,  # (64, 13)
            castling_feat_expanded  # (64, 4)
        ], axis=1)

        x = torch.tensor(nodes_feat, dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        data = Data(x=x, edge_index=self.edge_index, y=y)

        return data
