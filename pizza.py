"""
pizza: array rows x cols con caratteri ASCII (M o T)
min_ingredients: minimo numero di M e T in ogni slice
max_slice_size: massima dimensione di ogni slice

punteggio totale: rows x cols

slice: tupla (r_1, c_1, r_2, c_2) 
"""

import numpy as np

from itertools import product

def get_kernel_sizes(min_ingredients_per_slice, max_slice_dim):
    max_kernel_size = min_ingredients_per_slice + min_ingredients_per_slice
    x = 1
    y = max_kernel_size
    kernel_sizes = []

    for x, y in product(range(1, max_slice_dim + 1), range(1, max_slice_dim  + 1)):
        if x * y >= max_kernel_size and x * y <= max_slice_dim:
            kernel_sizes.append((x, y))

    return kernel_sizes

def possible_kernels(filename):
    if filename == "example.in":
        return [(1, 2), (2, 1)]
    elif filename == "small.in":
        return [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (1, 4), (4, 1), (1, 5), (5, 1)]
    elif filename == "medium.in":
        return [(12, 1), (12, 1), (1, 11), (11, 1), (1, 10), (10, 1), (9, 1), (1, 9), (4, 2), (2, 4), (1, 8), (8, 1), (3, 3)]
    elif filename == "big.in":
        return [(1, 14), (14, 1), (2, 7), (7, 2), (1, 13), (13, 1), (6, 2), (2, 6), (1, 12), (12, 1), (3, 4), (4, 3)]


def parse_input_file(file_name):
    f = open(file_name)

    line = f.readline()
    con = line.split(' ')
    r = int(con[0])
    c = int(con[1])
    
    min_t = int(con[2])     # minimum number of tomatoes and blabla
    max_size = int(con[3])  # max slice size

    matrix = np.zeros(shape=(r, c))
    m_i = 0
    for line in f:
        if not line.startswith(str(r)):
            indices = [i for i, ltr in enumerate(line) if ltr == 'T']
            matrix[m_i][indices] = 1
            m_i += 1
    f.close()
    return matrix, min_t, max_size

def calculate_score(kernel_matrix):
    score = 0
    for line in kernel_matrix:
        score += (abs(line[0] - line[2]) + 1) * (abs(line[1] - line[3]) + 1)
    return score

# creo una lista lunga m x n inizializzata a False che conterra' True se la cella i-esima e' presa da uno slice
# per ogni coppia (s, q) di kernel possibili
#     partendo da (0, 0) considero la cella (i, j), incrementando di 1 prima i fino a m, e poi j fino a n
#         per ogni cella nello slice
#             controllo che nella lista sia a False
#         se nessuna cella era a True
#             controllo che il numero minimo di ingredienti L sia rispettato sia per i Tomato che per i Mushroom
#                 in questo caso aggiungo questo slice e' valido e lo aggiungo alla lista

def slice_is_not_taken(occupied_matrix, kernel, row, col):
    kernel_row_size, kernel_col_size = kernel
    submatrix = occupied_matrix[row:row + kernel_row_size, col:col + kernel_col_size]
    all_false = submatrix == False
    return all_false.all()

def find_starting_slices(pizza, starting_kernels, MIN_T):
    starting_slices = []
    rows, cols = pizza.shape
    occupied_matrix = np.zeros(pizza.shape, dtype=bool)
    index = 0
    for kernel in starting_kernels:
        index += 1
        print(str(index * 100 /len(starting_kernels)) + '%')
        for row, col in product(range(rows), range(cols)):
            if slice_is_not_taken(occupied_matrix, kernel, row, col):
                if row + kernel[0] < rows and col + kernel[1] < cols:
                    # conta uno, sottrai al max e sai zero
                    my_slice = pizza[row:row + kernel[0], col:col + kernel[1]]
                    one_num = my_slice.sum()
                    zero_num = (kernel[0] * kernel[1]) - one_num
                    if one_num >= MIN_T and zero_num >= MIN_T:
                        starting_slices.append((row, col, row + kernel[0] - 1, col + kernel[1] - 1))
                        occupied_matrix[row:row + kernel[0], col:col + kernel[1]] = True
    return starting_slices, occupied_matrix


def final_score(FILE):
    pizza, MIN_T, MAX_SIZE = parse_input_file(FILE)
    MAX_SCORE = pizza.shape[0] * pizza.shape[1]
    starting_kernels = possible_kernels(FILE)
    # starting_kernels = possible_kernels(FILE)
    starting_slices, occupied_matrix = find_starting_slices(pizza, starting_kernels, MIN_T)
    # pprint(starting_slices)
    score = calculate_score(starting_slices)
    return score, MAX_SCORE

# dati gli slice iniziali espanderli finché si può lungo le righe e lungo le colonne

FILES = 'small.in'
FILEM = 'medium.in'
FILEB = 'big.in'

from pprint import pprint

s1, m1 = final_score(FILES)
s2, m2 = final_score(FILEM)
s3, m3 = final_score(FILEB)
MAX_SCORE = m1 + m2 + m3
score = s1 + s2 + s3
print('max score {}, score {} (percentage {})'.format(MAX_SCORE, score, score * 100. / MAX_SCORE))

