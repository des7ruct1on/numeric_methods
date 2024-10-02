import numpy as np



# Метод Гаусса с выбором главного элемента (частичным выбором главного элемента)
def gauss_elimination_with_pivot(A, b):
    n = len(A)
    A = A.astype(float)  # Преобразуем матрицу к float для избежания ошибок с типами
    b = b.astype(float)

    # Прямой ход: Приведение матрицы к верхнетреугольной форме с выбором главного элемента
    for i in range(n):
        # Выбор главного элемента по столбцу i
        max_row = np.argmax(np.abs(A[i:, i])) + i

        # Если максимальный элемент не на диагонали, меняем строки местами
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]

        # Приведение элементов ниже диагонали к нулю
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j][i:] = A[j][i:] - factor * A[i][i:]
            b[j] = b[j] - factor * b[i]

    # Обратный ход: решение системы методом обратной подстановки
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]

    return x


# Функция решения через библиотечный метод numpy для проверки
def solve_numpy(A, b):
    return np.linalg.solve(A, b)


def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        n = int(file.readline().strip())  # Считываем размерность
        matrix = []
        for _ in range(n):
            row = list(map(float, file.readline().strip().split()))
            matrix.append(row)
        vector = list(map(float, file.readline().strip().split()))

    return np.array(matrix), np.array(vector)


# Основная программа

filename = "input.txt"

A, b = read_matrix_from_file(filename)
# Решение системы через метод Гаусса с выбором главного элемента
solution_gauss = gauss_elimination_with_pivot(A.copy(), b.copy())

# Решение системы через библиотечный метод numpy
solution_numpy = solve_numpy(A.copy(), b.copy())

# Сравнение решений
print("Решение системы, метод Гаусса с выбором главного элемента, вектор x:", solution_gauss)
print("Решение системы, используя numpy вектор x:", solution_numpy)

# Вычисление ошибки
error = np.abs(solution_gauss - solution_numpy)
print("Абсолютная ошибка между решениями:", error)
