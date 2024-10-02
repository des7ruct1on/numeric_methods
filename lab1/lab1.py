import numpy as np

def det(x):
    if (x.shape[0] != x.shape[1]):
        raise ValueError("Матрица должна быть квадратной")
    if (x.shape[0] == 2 and x.shape[1] == 2):
        return x[0][0] * x[1][1] - x[1][0] * x[0][1]
    elif (x.shape[0] == 1 and x.shape[1] == 1):
        return x[0][0]
    
    res = 0 
    for i in range(x.shape[0]):
        sub_matrix = np.delete(np.delete(x, i, axis=0), 0, axis=1)
        res += ((-1) ** i) * x[i][0] * det(sub_matrix)
        
    return res


def method_progonki(matrix, vector):
    n = len(vector)  # Размер системы

    # Проверка на соответствие размерностей
    if matrix.shape[0] != n or matrix.shape[1] != n:
        raise ValueError("Матрица должна быть квадратной и совпадать с размерностью вектора.")
    
    # Проверка на трёхдиагональность
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and matrix[i][j] != 0:
                raise ValueError("Матрица должна быть трёхдиагональной.")

    # Проверка на нулевую главную диагональ
    if any(matrix[i][i] == 0 for i in range(n)):
        raise ValueError("Элементы главной диагонали не должны быть равны нулю.")

    # Проверка на ненулевой определитель
    if det(matrix) == 0:
        raise ValueError("Определитель матрицы равен нулю. Система не имеет единственного решения.")

    # Извлекаем поддиагональ, главную диагональ и наддиагональ из матрицы
    a = [0] + [matrix[i][i - 1] for i in range(1, n)]  # Поддиагональ 
    b = [matrix[i][i] for i in range(n)]               # Главная диагональ
    c = [matrix[i][i + 1] for i in range(n - 1)] + [0] # Наддиагональ 

    # Прямой ход
    alpha = [0] * n
    beta = [0] * n

    alpha[1] = -c[0] / b[0]
    beta[1] = vector[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] + a[i] * alpha[i]
        alpha[i + 1] = -c[i] / denom
        beta[i + 1] = (vector[i] - a[i] * beta[i]) / denom

    # Обратный ход
    x = [0] * n
    x[-1] = (vector[-1] - a[-1] * beta[-1]) / (b[-1] + a[-1] * alpha[-1])

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]

    return x

def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        n = int(file.readline().strip())  # Считываем размерность
        matrix = []
        for _ in range(n):
            row = list(map(float, file.readline().strip().split()))
            matrix.append(row)
        vector = list(map(float, file.readline().strip().split()))
    
    return np.array(matrix), np.array(vector)

filename = '../lab2/input.txt'
try:
    A, b = read_matrix_from_file(filename)
    solution = method_progonki(A, b)
    print("Решение системы:", solution)
except ValueError as e:
    print("Ошибка:", e)

try:
    x = np.linalg.solve(A, b)
    print("Решение системы 2:", x)
except np.linalg.LinAlgError as e:
    print("Ошибка при решении системы:", e)