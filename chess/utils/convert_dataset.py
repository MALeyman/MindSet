import pandas as pd
import numpy as np
import chess  


def index_to_move(index):
    '''  
    Ход из индекса
    '''
    from_sq = index // 64
    to_sq = index % 64
    file = lambda x: chr((x % 8) + ord('a'))
    rank = lambda x: str((x // 8) + 1)
    move_str = file(from_sq) + rank(from_sq) + file(to_sq) + rank(to_sq)
    return move_str



def move_to_index(move_str):
    ''' 
    Индекс из строкового значения хода
    '''
    file = lambda c: ord(c) - ord('a')
    rank = lambda c: int(c) - 1
    from_sq = rank(move_str[1]) * 8 + file(move_str[0])
    to_sq = rank(move_str[3]) * 8 + file(move_str[2])
    return from_sq * 64 + to_sq




# ================    Обработка (преобразование) датасета    =============================


# --------------     Конвертирование датасета в кратком виде  ----------------------
   
piece_to_num = {'P':1,'N':2,'B':3,'R':4,'Q':5,'K':6,'p':7,'n':8,'b':9,'r':10,'q':11,'k':12}

def fen_to_feature_vector(fen):
    board = chess.Board(fen)
    pos_vector = [0]*64
    for sq, piece in board.piece_map().items():
        pos_vector[sq] = piece_to_num[piece.symbol()]
    turn = 1 if board.turn else 0
    castling_rights = [
        1 if board.has_kingside_castling_rights(chess.WHITE) else 0,
        1 if board.has_queenside_castling_rights(chess.WHITE) else 0,
        1 if board.has_kingside_castling_rights(chess.BLACK) else 0,
        1 if board.has_queenside_castling_rights(chess.BLACK) else 0,
    ]
    ep_square = board.ep_square if board.ep_square is not None else -1
    ep_vector = [0]*64
    if ep_square != -1:
        ep_vector[ep_square] = 1
    feature_vector = pos_vector + [turn] + castling_rights + ep_vector
    return feature_vector


def convert_dataset_with_info(input_csv, output_csv):
    ''' 
    Конвертирование датасета в кратком виде
    '''
    df = pd.read_csv(input_csv)
    data = []
    for _, row in df.iterrows():
        fen = row['fen']
        move = row['move']
        features = fen_to_feature_vector(fen)
        move_index = move_to_index(move)
        data.append(features + [move_index])
    cols = [f'pos_{i}' for i in range(64)] + ['turn', 'castle_K', 'castle_Q', 'castle_k', 'castle_q'] + [f'ep_{i}' for i in range(64)] + ['move_idx']
    df_out = pd.DataFrame(data, columns=cols)
    df_out.to_csv(output_csv, index=False)



# --------------     Конвертирование датасета в  onehot  ----------------------

piece_to_idx = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
                'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}

def fen_to_onehot_vector(fen):
    board = chess.Board(fen)
    onehot = np.zeros((12,8,8), dtype=np.int8)
    for sq, piece in board.piece_map().items():
        row, col = divmod(sq, 8)
        onehot[piece_to_idx[piece.symbol()], row, col] = 1

    castling = np.array([
        1 if board.has_kingside_castling_rights(chess.WHITE) else 0,
        1 if board.has_queenside_castling_rights(chess.WHITE) else 0,
        1 if board.has_kingside_castling_rights(chess.BLACK) else 0,
        1 if board.has_queenside_castling_rights(chess.BLACK) else 0,
    ], dtype=np.int8)

    ep = np.zeros((8,8), dtype=np.int8)
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        ep[row, col] = 1

    feature_vector = np.concatenate([
        onehot.flatten(),
        castling,
        ep.flatten()
    ])
    return feature_vector


def convert_dataset_to_onehot(input_csv, output_csv):
    ''' 
    Конвертирование  датасета в One hot
    '''
    df = pd.read_csv(input_csv)
    data = []
    for _, row in df.iterrows():
        fen = row['fen']
        move = row['move']
        features = fen_to_onehot_vector(fen)
        move_index = move_to_index(move)
        data.append(np.append(features, move_index))
    # Формируем названия колонок
    cols = [f'p{i}' for i in range(12*8*8)] + ['castle_K', 'castle_Q', 'castle_k', 'castle_q'] + [f'ep_{i}' for i in range(8*8)] + ['move_idx']
    df_out = pd.DataFrame(data, columns=cols)
    df_out.to_csv(output_csv, index=False)



# =====================                 Визуализация  доски               ===============================

#  --------------------             Вывод   доски  в текстовом формате           ---------------------------------------
channel_to_piece = {
    0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
    6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k'
}

def onehot_to_fen_and_text(board_tensor):
    """
    Преобразует тензор с one-hot кодировкой шахматной позиции в строку FEN (без порта хода и другой информации) 
    и в список списков символов, представляющих каждый ряд доски для удобного визуального вывода.

    Аргументы:
    board_tensor (torch.Tensor): тензор формы (1, 13, 8, 8), где первые 12 каналов — это one-hot слои для фигур.

    Возвращает:
    tuple:
        fen (str): строка позиции в формате FEN (первая часть — расположение фигур),
        board_text (list of list of str): двумерный список с символами для каждого поля ('P', 'n', '.' и т.д.).

    Логика:
    - Перебирает каждый ряд и столбец доски.
    - Определяет, какая фигура стоит на поле, основываясь на one-hot каналах.
    - Для пустых полей увеличивает счетчик пустых и записывает число в FEN.
    - Для фигур добавляет соответствующий символ.
    - Формирует и возвращает строку FEN и удобочитаемый текстовый формат доски.
    """
    # board_tensor shape: (1, 13, 8, 8)
    # первые 12 каналов для фигур
    onehot = board_tensor[0, :12].cpu().numpy()  # (12,8,8)
    fen_rows = []
    board_text = []
    for row in range(8):
        fen_row = ''
        empty_count = 0
        text_row = []
        for col in range(8):
            piece = '.'
            for ch in range(12):
                if onehot[ch, row, col] == 1:
                    piece = channel_to_piece[ch]
                    break
            if piece == '.':
                empty_count += 1
                text_row.append('.')
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece
                text_row.append(piece)
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
        board_text.append(text_row)
    fen = '/'.join(fen_rows)
    return fen, board_text



def print_board(board_text):
    """
    Печатает доску в текстовом формате из списка списков символов.

    Аргументы:
    board_text (list of list of str): двумерный список символов шахматной доски.

    Вывод:
    Консольный вывод доски в виде 8 строк по 8 символов, разделённых пробелами.
    """
    for row in board_text:
        print(' '.join(row))



# -------------------------           Вывод доски с видом шахматных фигур          --------------------------
channel_to_piece = {
    0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
    6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k'
}

# Карта фигур для отображения Unicode символов
piece_to_unicode = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
    '.': '·'  # точка для пустой клетки
}

def onehot_to_fen_and_text_unicode(board_tensor):
    """
    Преобразует тензор с one-hot кодировкой шахматной позиции в строку FEN (без дополнительной информации) 
    и в список списков символов для отображения доски с использованием Unicode символов шахматных фигур.

    Аргументы:
    board_tensor (torch.Tensor): тензор формы (1, 13, 8, 8), где первые 12 каналов — один-hot слои для фигур.

    Возвращает:
    tuple:
        fen (str): строка позиции в формате FEN (расположение фигур),
        board_text (list of list of str): двумерный список символов для отображения (используйте для печати с Unicode символами).

    Логика:
    - Для каждого поля ищет фигуру, исходя из one-hot каналов.
    - Считает подряд идущие пустые клетки, формирует FEN-секцию.
    - Формирует 8 строк FEN и 8 строк для текстового отображения.
    """
    onehot = board_tensor[0, :12].cpu().numpy()
    fen_rows = []
    board_text = []
    for row in range(8):
        fen_row = ''
        empty_count = 0
        text_row = []
        for col in range(8):
            piece = '.'
            for ch in range(12):
                if onehot[ch, row, col] == 1:
                    piece = channel_to_piece[ch]
                    break
            if piece == '.':
                empty_count += 1
                text_row.append('O')
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece
                text_row.append(piece)
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
        board_text.append(text_row)
    fen = '/'.join(fen_rows)
    return fen, board_text


def print_board_unicode(board_text):
    """
    Печатает шахматную доску в консоли с использованием Unicode символов шахматных фигур.

    Аргументы:
    board_text (list of list of str): двумерный список символов доски, как возвращается из onehot_to_fen_and_text_unicode.

    Вывод:
    Для каждого символа выводится соответствующий Unicode символ фигуры  или точка для пустых полей.
    """
    for row in board_text:
        print(' '.join(piece_to_unicode.get(p, p) for p in row))





def fix_orientation_and_colors(onehot):
    """
    Исправляет ориентацию и меняет цвета фигур в one-hot представлении шахматной доски.

    Аргументы:
    onehot (np.ndarray): массив формы (12, 8, 8) с one-hot кодировкой расположения фигур,
                        где первые 6 каналов — белые фигуры, следующие 6 — чёрные.

    Логика:
    - Отзеркаливает доску по вертикали, чтобы изменить ориентацию (разворот по вертикали).
    - Меняет местами каналы с белыми и чёрными фигурами — белые фигуры заменяются чёрными и наоборот.
      Это нужно для корректного отображения/анализа доски с точки зрения другой стороны.

    Возвращает:
    np.ndarray той же формы (12, 8, 8) с исправленной ориентацией и цветами фигур.
    """
    onehot_flipped = onehot[:, ::-1, :]
    # Меняем местами белые и черные фигуры
    fixed_onehot = np.zeros_like(onehot_flipped)
    fixed_onehot[0:6] = onehot_flipped[6:12]  # белые берутся из черных
    fixed_onehot[6:12] = onehot_flipped[0:6]  # черные берутся из белых
    return fixed_onehot



def onehot_to_board_text(onehot_12_8_8):
    """
    Преобразует исправленный one-hot тензор шахматной доски в удобочитаемый список символов фигур по полям.

    Аргументы:
    onehot_12_8_8 (np.ndarray): массив формы (12, 8, 8) с one-hot кодировкой фигур.

    Логика:
    - Вызывает fix_orientation_and_colors для корректировки ориентации и цветов.
    - Перебирает каждую клетку, определяет, какая фигура там стоит.
    - Если фигуры нет, ставит точку '.'.
    - Формирует двумерный список символов (8 строк по 8 символов).

    Возвращает:
    list of list of str — двумерный список символов фигур (например, 'P', 'n', '.' и т.д.) для визуализации.
    """
    fixed_onehot = fix_orientation_and_colors(onehot_12_8_8)
    board_text = []
    for row in range(8):
        text_row = []
        for col in range(8):
            piece = '.'
            for ch in range(12):
                if fixed_onehot[ch, row, col] == 1:
                    piece = channel_to_piece[ch]
                    break
            text_row.append(piece)
        board_text.append(text_row)
    return board_text


# Из матрицы в FEN
def onehot_to_board_fen(onehot_12_8_8):

    """
    Преобразует one-hot матрицу шахматной доски в строку позиции в формате FEN (Forsyth-Edwards Notation).

    Аргументы:
    onehot_12_8_8 (np.ndarray): матрица размера (12, 8, 8), где каждый из 12 каналов — один тип фигуры (6 белых и 6 черных),
                               представленная one-hot кодировкой на доске.

    Логика:
    - Сначала корректирует ориентацию и цвета фигур, вызвав fix_orientation_and_colors.
    - Для каждого ряда доски последовательно:
      - Определяет фигуру на каждой клетке (или пустоту).
      - Считает подряд идущие пустые клетки, используя цифру в FEN для сокращения записи.
      - Формирует строку для ряда с упрощением пустых клеток.
    - Объединяет все 8 строк через '/'.

    Возвращает:
    str — строку позиции в формате FEN, описывающую только расположение фигур (первая часть полной FEN-нотации).
    """

    fixed_onehot = fix_orientation_and_colors(onehot_12_8_8)
    fen_rows = []
    for row in range(8):
        fen_row = ''
        empty_count = 0
        for col in range(8):
            piece = None
            for ch in range(12):
                if fixed_onehot[ch, row, col] == 1:
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
    return '/'.join(fen_rows)



#  Применяет заданный ход к текущей позиции, заданной в one-hot формате и признаках рокировки, возвращает новую позицию в формате FEN и в виде Unicode шахматной доски
def apply_move_and_get_new_board(board_tensor, castling_tensor, turn, move_index):
    """
    Применяет заданный ход к текущей позиции, заданной в one-hot формате и признаках рокировки, 
    и возвращает новую позицию в формате FEN и в виде Unicode шахматной доски.

    Параметры:
    - board_tensor (torch.Tensor): тензор с положением фигур (размерность (1, 13, 8, 8),
      первые 12 каналов — фигуры, 13-й — en passant).
    - castling_tensor (torch.Tensor): тензор признаков прав на рокировку (4 элемента).
    - turn (int): чей ход (1 для белых, 0 для черных).
    - move_index (torch.Tensor или int): индекс хода, кодируемый как from_sq * 64 + to_sq.

    Логика:
    - Преобразует one-hot позиции в FEN строку с помощью функции onehot_to_board_fen.
    - Преобразует признаки рокировки в строку (KQkq или '-').
    - Преобразует слой en passant в строку с координатой или '-'.
    - Формирует полную FEN строку с текущей позицией, ходом, рокировками, взятием на проходе,
      и начальными значениями счетчиков полуходов и полного хода.
    - Создает объект chess.Board с этой FEN-позицией.
    - Преобразует индекс хода в строку хода в формате UCI.
    - Проверяет законность хода на доске, если ход не совсем точен — ищет в легальных ходах ход с теми же from и to квадратами (учитывая возможные варианты промоции и др.).
    - Если ход законен, делает ход на доске.
    - Возвращает обновленную позицию как строку FEN и в виде Unicode отображения доски.

    Исключения:
    - Если ход не легален и не найден среди возможных легальных ходов, вызывает ValueError.

    Возвращает:
    tuple: (строка FEN обновленной позиции, строковое Unicode представление позиции).
    """   

    onehot = board_tensor[0, :12].cpu().numpy()
    board_fen = onehot_to_board_fen(onehot)
    
    castling_arr = castling_tensor.squeeze().cpu().numpy()
    rights = ''
    if castling_arr[0] == 1: rights += 'K'
    if castling_arr[1] == 1: rights += 'Q'
    if castling_arr[2] == 1: rights += 'k'
    if castling_arr[3] == 1: rights += 'q'
    castling_str = rights if rights else '-'
    
    ep_layer = board_tensor[0, 12].cpu().numpy().flatten()
    idx = np.argmax(ep_layer)
    if ep_layer[idx] == 0:
        ep_str = '-'
    else:
        rank = 8 - (idx // 8)
        file = chr(idx % 8 + ord('a'))
        ep_str = f"{file}{rank}"
    
    turn_char = 'w' if turn == 1 else 'b'

    full_fen = f'{board_fen} {turn_char} {castling_str} {ep_str} 0 1'
    
    board = chess.Board(full_fen)

    # def index_to_move_label(move_int):
    #     from_sq = move_int // 64
    #     to_sq = move_int % 64
    #     file = lambda x: chr(ord('a') + (x % 8))
    #     rank = lambda x: str((x // 8) + 1)
    #     return file(from_sq) + rank(from_sq) + file(to_sq) + rank(to_sq)

    move_str = index_to_move_label(move_index.item())
    candidate_move = chess.Move.from_uci(move_str)

    if candidate_move in board.legal_moves:
        board.push(candidate_move)
    else:
        matched_move = None
        for m in board.legal_moves:
            if m.from_square == candidate_move.from_square and m.to_square == candidate_move.to_square:
                matched_move = m
                break
        if matched_move is None:
            raise ValueError(f"Move {move_str} is illegal in position {full_fen}")
        board.push(matched_move)

    return board.board_fen(), board.unicode()




import numpy as np

channel_to_piece = {
    0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
    6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k'
}


def index_to_move_label(move_int):
    from_sq = move_int // 64
    to_sq = move_int % 64
    file = lambda sq: chr(ord('a') + (sq % 8))
    rank = lambda sq: str((sq // 8) + 1)
    return file(from_sq) + rank(from_sq) + file(to_sq) + rank(to_sq)


def onehot_to_fen(onehot_13_8_8, castling_feat, move_index, turn_char='w'):
    onehot = onehot_13_8_8[:12]  # первые 12 слоев - фигуры (12,8,8)
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

    # Формируем строку рокировки
    rights = ''
    if castling_feat[0] == 1: rights += 'K'
    if castling_feat[1] == 1: rights += 'Q'
    if castling_feat[2] == 1: rights += 'k'
    if castling_feat[3] == 1: rights += 'q'
    castling_str = rights if rights else '-'

    # Определяем en passant из 13-го слоя onehot
    ep_layer = onehot_13_8_8[12].flatten()
    idx = np.argmax(ep_layer)
    if ep_layer[idx] == 0:
        ep_str = '-'
    else:
        rank = 8 - (idx // 8)
        file = chr(idx % 8 + ord('a'))
        ep_str = f"{file}{rank}"

    # Собираем полную FEN (без информации о счёте полуходов и номере полного хода)
    full_fen = f"{board_fen} {turn_char} {castling_str} {ep_str} 0 1"
    move_str = index_to_move_label(move_index.item())
    return full_fen, move_str

def get_fen_and_label(board_feat, castling_feat, label,  turn_char='w'):

    # Преобразуем в numpy
    board_feat_np = board_feat.numpy()
    castling_feat_np = castling_feat.numpy()

    # Получаем строку FEN позиции
    fen, move_str = onehot_to_fen(board_feat_np, castling_feat_np, label, turn_char)
    
    return fen, move_str


