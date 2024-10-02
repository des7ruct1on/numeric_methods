import numpy as np

# Функция для метода простых итераций
def simple_iteration(A, b, tol=1e-6, max_iterations=1000):
    n = len(b)  # количество переменных
    x = np.zeros_like(b)  # начальное приближение, вектор нулей
    x_new = np.zeros_like(b)  # новый вектор для обновленного решения
    B = np.zeros_like(A)  # матрица для преобразования
    c = np.zeros_like(b)  # вектор свободных членов

    # Преобразование системы к виду x = Bx + c
    for i in range(n):
        for j in range(n):
            if i != j:
                B[i, j] = -A[i, j] / A[i, i]
        c[i] = b[i] / A[i, i]

    # Итерационный процесс
    for iteration in range(max_iterations):
        x_new = B.dot(x) + c  # вычисляем новое приближение
        # Проверяем, достигнута ли заданная точность
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f'Метод простых итераций сошелся за {iteration} итераций')
            return x_new
        x = x_new  # обновляем старое приближение
    raise Exception('Метод простых итераций не сошелся')

# Функция для метода Зейделя (Гаусса-Зейделя)
def seidel(A, b, tol=1e-6, max_iterations=1000):
    n = len(b)  # количество переменных
    x = np.zeros_like(b)  # начальное приближение, вектор нулей

    # Итерационный процесс
    for iteration in range(max_iterations):
        x_new = np.copy(x)  # копируем старый вектор

        # Обновляем решение для каждой переменной
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))  # сумма для предыдущих переменных
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))  # сумма для последующих переменных
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]  # обновление переменной

        # Проверяем, достигнута ли заданная точность
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f'Метод Зейделя сошелся за {iteration} итераций')
            return x_new

        x = x_new  # обновляем старое приближение
    raise Exception('Метод Зейделя не сошелся')

def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        n = int(file.readline().strip())  # Считываем размерность
        matrix = []
        for _ in range(n):
            row = list(map(float, file.readline().strip().split()))
            matrix.append(row)
        vector = list(map(float, file.readline().strip().split()))

    return np.array(matrix), np.array(vector)


# Задаем точность и решаем СЛАУ
tolerance = 1e-6

A, b = read_matrix_from_file("input.txt")
# Solve using Simple Iteration Method
solution_simple = simple_iteration(A, b, tol=tolerance)
print("Решение через простые итерации:")
print(solution_simple)

# Solve using Seidel's Method
solution_seidel = seidel(A, b, tol=tolerance)
print("Решение через метод Сейделя:")
print(solution_seidel)

print("Решение через numpy:")
print(np.linalg.solve(A, b))
