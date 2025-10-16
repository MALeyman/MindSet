import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import chess
 


def count_moves_and_turns(csv_path, fen_column='fen', count=None):
    """
    Анализирует шахматные позиции в формате FEN из CSV-файла и подсчитывает:
    - общее количество сделанных ходов белыми и черными,
    - количество позиций, в которых сейчас ходят белые и черные.

    Параметры:
    csv_path (str): путь к CSV-файлу с колонкой, содержащей FEN-строки.
    fen_column (str): имя колонки с FEN-позициями (по умолчанию 'fen').
    count (int или None): если задано число, выводит в консоль указанное количество позиций и их досок для отладки.

    Возвращает:
    tuple: (white_move_count, black_move_count, white_turn_count, black_turn_count)
    где
    white_move_count (int): суммарное количество ходов, сделанных белыми,
    black_move_count (int): суммарное количество ходов, сделанных черными,
    white_turn_count (int): количество позиций, где ход белых,
    black_turn_count (int): количество позиций, где ход черных.
    """
    df = pd.read_csv(csv_path)
    white_move_count = 0
    black_move_count = 0
    white_turn_count = 0
    black_turn_count = 0
    board = None
    fen = None
    for fen in df[fen_column]:

        parts = fen.split(' ')
        turn = parts[1]
        full_move = int(parts[5])

        # Счёт сделанных ходов
        if turn == 'w':
            white_moves = full_move - 1
            black_moves = full_move - 1
            white_turn_count += 1
        else:
            white_moves = full_move
            black_moves = full_move - 1
            black_turn_count += 1
        white_move_count += white_moves
        black_move_count += black_moves
        if not count==None:
            print(fen)
            board = chess.Board(fen)
            print(board)
            count -= 1
            if count < 1:
                break

    return white_move_count, black_move_count, white_turn_count, black_turn_count, board, fen  


 # выводит статистику по отдельным полям FEN-позиций из CSV-файла.
def analyze_fen_fields(csv_path, fen_column='fen'):
    """
    Анализирует и выводит статистику по отдельным полям FEN-позиций из CSV-файла.

    Параметры:
    csv_path (str): путь к CSV-файлу с колонкой, содержащей FEN-строки.
    fen_column (str): имя колонки с FEN-позициями (по умолчанию 'fen').

    Для каждой позиции в колонке извлекаются и собираются статистики по полям FEN:
    - Очередь хода (turn): кто сейчас ходит ('w' или 'b').
    - Права на рокировку (castling): строка с правами (например, 'KQkq').
    - Взятие на проходе (en passant): квадрат для взятия на проходе или '-'.
    - Счётчик полуходов (halfmove clock): количество полуходов без взятия и пешечного хода.
    - Номер полного хода (fullmove number): номер хода партии.

    Результат:
    Печатает частоты появления каждого уникального значения для перечисленных полей.
    Пропускает некорректные записи FEN (с недостаточным числом полей).
    """

    df = pd.read_csv(csv_path)
    
    turn_counter = Counter()
    castling_counter = Counter()
    en_passant_counter = Counter()
    halfmove_counter = Counter()
    fullmove_counter = Counter()
    
    for fen in df[fen_column]:
        parts = fen.split(' ')
        if len(parts) < 6:
            continue  
        
        turn = parts[1]
        castling = parts[2]
        en_passant = parts[3]
        halfmove = parts[4]
        fullmove = parts[5]
        
        turn_counter[turn] += 1
        castling_counter[castling] += 1
        en_passant_counter[en_passant] += 1
        halfmove_counter[halfmove] += 1
        fullmove_counter[fullmove] += 1
    
    print("Очередь хода (turn):")
    for k,v in turn_counter.most_common():
        print(f"  {k}: {v}")
    
    print("\nПрава на рокировку (castling):")
    for k,v in castling_counter.most_common():
        print(f"  '{k}': {v}")
    
    print("\nВзятие на проходе (en passant):")
    for k,v in en_passant_counter.most_common():
        print(f"  '{k}': {v}")
    
    print("\nСчётчик полуходов (halfmove clock):")
    for k,v in sorted(halfmove_counter.items(), key=lambda x: int(x[0])):
        print(f"  {k}: {v}")

    print("\nНомер полного хода (fullmove number):")
    for k,v in sorted(fullmove_counter.items(), key=lambda x: int(x[0])):
        print(f"  {k}: {v}")





# Графический анализ датасета
def annotate_bars(ax):
    ''' 
    Графический анализ данных
    '''
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, rotation=0, xytext=(0, 3),
                    textcoords='offset points')

def plot_fen_field_distribution_with_values(csv_path, fen_column='fen'):
    df = pd.read_csv(csv_path)

    turn_list = []
    castling_list = []
    en_passant_list = []
    halfmove_list = []
    fullmove_list = []

    for fen in df[fen_column]:
        parts = fen.split(' ')
        if len(parts) < 6:
            continue
        turn_list.append(parts[1])
        castling_list.append(parts[2])
        en_passant_list.append(parts[3])
        halfmove_list.append(int(parts[4]))
        fullmove_list.append(int(parts[5]))

    plt.figure(figsize=(14,12))

    plt.subplot(3,2,1)
    ax1 = sns.countplot(x=turn_list)
    plt.title('Очередь хода (turn)')
    plt.xlabel('Цвет хода')
    plt.ylabel('Количество')
    annotate_bars(ax1)

    plt.subplot(3,2,2)
    castling_counts = Counter(castling_list)
    castling_items = list(castling_counts.items())
    castling_labels = [item[0] for item in castling_items]
    castling_values = [item[1] for item in castling_items]
    ax2 = sns.barplot(x=castling_labels, y=castling_values)
    plt.title('Права на рокировку')
    plt.xlabel('Права рокировки')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    annotate_bars(ax2)

    plt.subplot(3,2,3)
    en_passant_counts = Counter(en_passant_list)
    en_passant_items = en_passant_counts.most_common(10)
    en_passant_labels = [item[0] for item in en_passant_items]
    en_passant_values = [item[1] for item in en_passant_items]
    ax3 = sns.barplot(x=en_passant_labels, y=en_passant_values)
    plt.title('Возможности взятия на проходе (топ-10)')
    plt.xlabel('Поле')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    annotate_bars(ax3)

    plt.subplot(3,2,4)
    ax4 = sns.histplot(halfmove_list, bins=30, kde=False)
    plt.title('Счётчик полуходов (halfmove clock)')
    plt.xlabel('Количество полуходов с последнего хода пешки или взятия')
    plt.ylabel('Частота')
    
    for p in ax4.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width() / 2
        ax4.text(x, height, f'{int(height)}', ha='center', va='bottom', fontsize=8)

    plt.subplot(3,2,5)
    ax5 = sns.histplot(fullmove_list, bins=30, kde=False)
    plt.title('Номер полного хода (fullmove number)')
    plt.xlabel('Номер полного хода')
    plt.ylabel('Частота')
    
    for p in ax5.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width() / 2
        ax5.text(x, height, f'{int(height)}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()
