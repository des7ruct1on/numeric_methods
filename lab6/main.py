import numpy as np
import math
import numpy as np
from scipy.optimize import fsolve

# --- Определим функции для системы уравнений ---
def f1(x1, x2):
    return x1 - math.cos(x2) - 2

def f2(x1, x2):
    return x2 - math.sin(x1) - 2

# --- Матрица Якоби (для метода Ньютона) ---
def jacobian(x1, x2):
    df1_dx1 = 1
    df1_dx2 = math.sin(x2)
    df2_dx1 = -math.cos(x1)
    df2_dx2 = 1
    return np.array([[df1_dx1, df1_dx2], [df2_dx1, df2_dx2]])

# --- Метод Ньютона для решения системы ---
def newton_system(x0, y0, tol=1e-6, max_iter=100):
    x, y = x0, y0
    for i in range(max_iter):
        # Значения функций в текущей точке
        F = np.array([f1(x, y), f2(x, y)])
        
        # Матрица Якоби и обратная матрица
        J = jacobian(x, y)
        J_inv = np.linalg.inv(J)
        
        # Улучшаем приближение
        delta = np.dot(J_inv, -F)
        x, y = x + delta[0], y + delta[1]
        
        # Проверяем сходимость
        if np.linalg.norm(delta) < tol:
            return (x, y), i  # Возвращаем решение и число итераций
    
    raise ValueError("Метод Ньютона не сошелся за отведенное число итераций")

# --- Метод простой итерации ---
def simple_iteration_system(x0, y0, tol=1e-6, max_iter=100):
    def phi1(x2):
        return math.cos(x2) + 2

    def phi2(x1):
        return math.sin(x1) + 2

    x, y = x0, y0
    for i in range(max_iter):
        x_new = phi1(y)
        y_new = phi2(x)

        if abs(x_new - x) < tol and abs(y_new - y) < tol:
            return (x_new, y_new), i  # Возвращаем решение и число итераций

        x, y = x_new, y_new

    raise ValueError("Метод простой итерации не сошелся за отведенное число итераций")

# --- Основная программа ---
x0, y0 = 0.5, 0.5  # Начальные приближения

def system(vars):
    x1, x2 = vars  # Переменные системы
    eq1 = x1 - np.cos(x2) - 2
    eq2 = x2 - np.sin(x1) - 2
    return [eq1, eq2]

# Начальное приближение
initial_guess = [0.5, 0.5]

# Решаем систему с помощью fsolve
solution = fsolve(system, initial_guess)

# Выводим решение
print(f"Решение системы scipy: x1 = {solution[0]}, x2 = {solution[1]}")

try:
    # Метод Ньютона
    root_newton, iter_newton = newton_system(x0, y0)
    print(f"Метод Ньютона: решение = {root_newton}, итерации = {iter_newton}")

    # Метод простой итерации
    root_simple, iter_simple = simple_iteration_system(x0, y0)
    print(f"Метод простой итерации: решение = {root_simple}, итерации = {iter_simple}")

except ValueError as e:
    print(e)
