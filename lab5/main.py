import numpy as np
import math

# Определяем основную функцию и её производную
def f(x):
    return math.exp(-x) + 3 * x**2 - 2 * x - 3

def f_prime(x):
    return -math.exp(-x) + 6 * x - 2

# --- 1. Метод бисекции (дихотомии) ---
def bisection_method(a, b, tol=1e-6):
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) и f(b) должны иметь разные знаки (функция должна менять знак на концах интервала).")
    
    iteration = 0
    while (b - a) / 2 > tol:
        iteration += 1
        c = (a + b) / 2  # середина интервала
        if f(c) == 0:  # нашли точное решение
            return c, iteration
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, iteration

# --- 2. Метод Ньютона ---
def newton_method(x0, tol=1e-6, max_iter=100):
    iteration = 0
    while iteration < max_iter:
        if abs(f_prime(x0)) < 1e-12:  # Проверка малой производной
            raise ValueError("Производная близка к нулю, метод Ньютона не применим.")
        
        x1 = x0 - f(x0) / f_prime(x0)  # Формула Ньютона
        if abs(x1 - x0) < tol:
            return x1, iteration
        x0 = x1
        iteration += 1
    raise ValueError("Метод Ньютона не сошелся за отведенное число итераций.")

# --- 3. Метод секущих ---
def secant_method(x0, x1, tol=1e-6, max_iter=100):
    iteration = 0
    while iteration < max_iter:
        f_x0 = f(x0)
        f_x1 = f(x1)
        if abs(f_x1 - f_x0) < 1e-12:  # Проверка на деление на ноль
            raise ValueError("Разность значений функции слишком мала, метод секущих не применим.")
        
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)  # Формула секущих
        if abs(x2 - x1) < tol:
            return x2, iteration
        x0, x1 = x1, x2
        iteration += 1
    raise ValueError("Метод секущих не сошелся за отведенное число итераций.")

# --- 4. Метод простой итерации ---
def simple_iteration_method(x0, tol=1e-6, max_iter=100):
    def phi(x):
        return (math.exp(-x) + 3 * x**2 - 3) / 2  # Итерационная функция

    def phi_prime(x):
        return -0.5 * math.exp(-x) + 3 * x  # Производная итерационной функции

    # Проверка условия сходимости |phi'(x0)| < 1
    if abs(phi_prime(x0)) >= 1:
        raise ValueError("Условие сходимости не выполняется: |phi'(x0)| >= 1.")

    iteration = 0
    while iteration < max_iter:
        x1 = phi(x0)  # Применяем итерационную функцию
        if abs(x1 - x0) < tol:
            return x1, iteration
        x0 = x1
        iteration += 1
    raise ValueError("Метод простой итерации не сошелся за отведенное число итераций.")

# --- Основная часть программы ---
a, b = 0, 2  # Интервал для метода бисекции
x0 = 1.0  # Начальное приближение
x1 = 1.5  # Второе приближение для метода секущих

try:
    root_bisection, iter_bisection = bisection_method(a, b)
    print(f"Метод бисекции: корень = {root_bisection}, итерации = {iter_bisection}")

    root_newton, iter_newton = newton_method(x0)
    print(f"Метод Ньютона: корень = {root_newton}, итерации = {iter_newton}")

    root_secant, iter_secant = secant_method(x0, x1)
    print(f"Метод секущих: корень = {root_secant}, итерации = {iter_secant}")

    root_simple, iter_simple = simple_iteration_method(x0)
    print(f"Метод простой итерации: корень = {root_simple}, итерации = {iter_simple}")

except ValueError as e:
    print(e)
