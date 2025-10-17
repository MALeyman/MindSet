import chess
import chess.svg
from IPython.display import SVG, display
import numpy as np
import torch

channel_to_piece = {
    0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
    6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k'
}

def onehot_to_fen(onehot_13_8_8, castling_feat, turn_char='w'):
    """
    Преобразует представление шахматной позиции из one-hot формата в строку в нотации FEN.

    Параметры:
    onehot_13_8_8 (np.array): Тензор размером (13, 8, 8), где первые 12 слоев – one-hot кодирование положения фигур на доске,
                              13-й слой – признак шаха на проходе (en passant).
    castling_feat (np.array): Массив из 4 элементов, содержащий признаки наличия прав рокировки для белых и чёрных:
                             [белая короткая, белая длинная, чёрная короткая, чёрная длинная].
    turn_char (str): Символ, обозначающий очередность хода ('w' для белых, 'b' для чёрных). По умолчанию 'w'.

    Возвращаемое значение:
    str: Полная строка шахматной позиции в формате FEN, включающая расположение фигур, очередность хода,
         права на рокировку, поле для взятия на проходе, и счетчики ходов (эти последний два в функции заданы фиксировано).

    Описание:
    - Функция проходит по слоям one-hot кода, преобразуя их в стандартное текстовое обозначение пешек, коней, слонов, ладей, ферзей и королей для белых и чёрных.
    - Опустошённые клетки кодируются цифрой, обозначающей количество пустых подряд идущих клеток.
    - Генерируется стандартная FEN-строка с разделителем '/' для рядов доски, начиная с 8-го ряда вниз.
    - Права рокировки формируются из массива флагов, отсутствующие права обозначаются '-'.
    - Поле для взятия на проходе вычисляется из 13-го слоя one-hot, если оно отсутствует, ставится '-'.
    - Счетчики ходов в конце FEN заданы как "0 1" по умолчанию.

    """
    onehot = onehot_13_8_8[:12]
    fen_rows = []
    for row in range(7, -1, -1):  
        fen_row = ''
        empty_count = 0
        for col in range(8):
            piece = None
            for ch in range(12):
                if onehot[ch, row, col] == 1:
                    piece = channel_to_piece[ch]
                    break
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    board_fen = '/'.join(fen_rows)

    rights = ''
    if castling_feat[0] == 1: rights += 'K'
    if castling_feat[1] == 1: rights += 'Q'
    if castling_feat[2] == 1: rights += 'k'
    if castling_feat[3] == 1: rights += 'q'
    castling_str = rights if rights else '-'

    ep_layer = onehot_13_8_8[12].flatten()
    idx = np.argmax(ep_layer)
    if ep_layer[idx] == 0:
        ep_str = '-'
    else:
        rank = 8 - (idx // 8)
        file = chr(idx % 8 + ord('a'))
        ep_str = f"{file}{rank}"

    return f"{board_fen} {turn_char} {castling_str} {ep_str} 0 1"


def index_to_move_label(move_int):
    """
    Преобразует целочисленный индекс шахматного хода в строковое обозначение хода.

    Параметры:
    move_int (int): Индекс хода, закодированный как число от 0 до 4095, где
                    целая часть от деления на 64 — это индекс начальной клетки (from_square),
                    а остаток от деления на 64 — индекс конечной клетки (to_square).

    Возвращаемое значение:
    str: Строка с ходом в формате  "e2e4", где первые две буквы и цифры — начальная клетка,
         а последние две — конечная клетка.
    """
    from_sq = move_int // 64
    to_sq = move_int % 64
    file = lambda sq: chr(ord('a') + (sq % 8))
    rank = lambda sq: str((sq // 8) + 1)
    return file(from_sq) + rank(from_sq) + file(to_sq) + rank(to_sq)


def display_positions_after_moves(board_tensor, castling_tensor, label, predicted_move_index, turn_char='w'):
    ''' 
    Вывод досок:
        Исходная позиция
        Позиция после предсказанного  хода
        Позиция после реального  хода
    '''
    # Преобразование input в numpy
    board_np = board_tensor.numpy()
    castling_np = castling_tensor.numpy()
    
    # Исходная позиция
    fen_start = onehot_to_fen(board_np, castling_np, turn_char)
    board_start = chess.Board(fen_start)
    
    # Позиция после предсказанного хода
    pred_move_str = index_to_move_label(predicted_move_index)
    board_pred = board_start.copy()
    move_pred = chess.Move.from_uci(pred_move_str)
    if move_pred in board_pred.legal_moves:
        board_pred.push(move_pred)
    else:
        print(f"Предсказанный ход {pred_move_str} не легален")
    
    # Позиция после реального хода label
    real_move_str = index_to_move_label(label.item() if isinstance(label, torch.Tensor) else label)
    board_real = board_start.copy()
    move_real = chess.Move.from_uci(real_move_str)
    if move_real in board_real.legal_moves:
        board_real.push(move_real)
    else:
        print(f"Реальный ход {real_move_str} не легален")
    
    # Отобразить доски
    print("-"*32)
    print("Исходная позиция:")
    display(SVG(chess.svg.board(board_start, size=350)))
    print("Позиция после предсказанного хода:")
    display(SVG(chess.svg.board(board_pred, size=350)))
    print("Позиция после реального хода:")
    display(SVG(chess.svg.board(board_real, size=350)))






import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.convert_dataset import index_to_move
from utils.convert_dataset import  get_fen_and_label

def predict_move(model, board, castling, label):
    ''' 
    Исходную позицию
    Реальный ход
    Вероятности ходов
    Индекс предсказанного хода
    Выводит доску с текущей позицией

    '''
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)
    board3 = board.unsqueeze(0)      
    castling3 = castling.unsqueeze(0)
    logits = model(board3, castling3)

    print("Логиты:")
    print(logits.shape) 
    print(logits)

    fen_str, move_label = get_fen_and_label(board, castling, label,  turn_char='w')

    print("\nИсходная позиция:")
    print(fen_str)
    print("\nРеальный ход:")
    print(move_label)


    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    print("\nВероятности:")
    print(probs)
    legal_move_indices, board1 = get_legal_moves(fen_str)  # Вывод доски, индексы возможных ходов
    # вероятности хода
    probs_1d = probs[0]
    legal_probs = probs_1d[legal_move_indices]
    best_idx = legal_move_indices[np.argmax(legal_probs)]
    print("\nИндекс предсказанного хода:")
    print(best_idx)
    # ход из индекса
    index_to_move(best_idx)
    
    return best_idx, fen_str, move_label



import chess
import chess.svg
from IPython.display import SVG, display
from utils.convert_dataset import move_to_index

def  get_legal_moves(fen):
    ''' 
    Все возможные ходы в данной позиции
    Вывод доски
    '''
    board1 = chess.Board(fen)
    legal_moves = list(board1.legal_moves)
    legal_moves_str = [move.uci() for move in legal_moves]

    print("Возможные ходы:")
    print(legal_moves_str)
    # Индексы возможных ходов
    legal_move_indices = [move_to_index(move.uci()) for move in legal_moves]
    print("Индексы возможных ходов:")
    print(legal_move_indices)
    svg_image = chess.svg.board(board1, size=450)
    display(SVG(svg_image))

    return legal_move_indices, board1









