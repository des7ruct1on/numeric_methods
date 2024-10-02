from array import array

import numpy as np


# Функция для поиска максимального внедиагонального элемента
def max_offdiag(A):
    n = A.shape[0]
    max_val = 0
    p, q = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > abs(max_val):
                max_val = A[i, j]
                p, q = i, j
    return max_val, p, q


# Функция для метода Якоби (метод вращений)
def jacobi_method(A, tol=1e-6, max_iterations=100):
    n = A.shape[0]
    V = np.eye(n)  # матрица собственных векторов
    iterations = 0

    for _ in range(max_iterations):
        # Поиск максимального внедиагонального элемента
        max_val, p, q = max_offdiag(A)

        # Если максимальный внедиагональный элемент меньше точности, завершаем
        if abs(max_val) < tol:
            break

        # Вычисление угла вращения
        if A[p, p] == A[q, q]:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan(2 * A[p, q] / (A[p, p] - A[q, q]))

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Создаем матрицу вращения
        J = np.eye(n)
        J[p, p] = cos_phi
        J[q, q] = cos_phi
        J[p, q] = -sin_phi
        J[q, p] = sin_phi

        # Обновляем матрицу A и матрицу собственных векторов V
        A = J.T @ A @ J
        V = V @ J

        iterations += 1

    # Собственные значения - это диагональные элементы матрицы A
    eigenvalues = np.diag(A)
    eigenvectors = V

    print(f"Количество итераций: {iterations}")

    return eigenvalues, eigenvectors

def read_file(filename: str) -> np.array:
    with open(filename, 'r') as file:
        n = int(file.readline().strip())  # Считываем размерность
        matrix = []
        for _ in range(n):
            row = list(map(float, file.readline().strip().split()))
            matrix.append(row)

    return np.array(matrix)

A = read_file("input.txt")

# Вызываем функцию для нахождения собственных значений и векторов с помощью метода Якоби
eigenvalues_jacobi, eigenvectors_jacobi = jacobi_method(A.copy(), tol=1e-6)

# Вывод результатов метода Якоби
print("Собственные значения (метод Якоби):")
print(eigenvalues_jacobi)
print("Собственные векторы (метод Якоби):")
print(eigenvectors_jacobi)

# Используем встроенную функцию NumPy для нахождения собственных значений и векторов
eigenvalues_np, eigenvectors_np = np.linalg.eig(A)

# Вывод результатов с использованием NumPy
print("\nСобственные значения numpy:")
print(eigenvalues_np)
print("Собственные векторы numpy:")
print(eigenvectors_np)
