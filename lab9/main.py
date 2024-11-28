import numpy as np
import matplotlib.pyplot as plt

# Определяем функцию и её производные
def func(x):
    return 1.0 / (x**3 + 64)

def second_derivative(x):
    numerator = -6 * x / (x**3 + 64)**3
    denominator = (x**3 + 64)
    return numerator / denominator

def fourth_derivative(x):
    numerator = (
        120 * x**4 - 960 * x**2 + 384
    )  # Это приближенное выражение для сложной функции, его нужно уточнять
    denominator = (x**3 + 64)**5
    return numerator / denominator

# Методы численного интегрирования
def rectangle_method(x0, x1, h):
    xi = np.arange(x0, x1, h)
    return h * np.sum(func((xi + xi + h) * 0.5))

def trapezoidal_method(x0, x1, h):
    xi = np.arange(x0, x1 + h, h)
    return h * (np.sum(func(xi)) - 0.5 * (func(x0) + func(x1)))

def simpson_method(x0, x1, h):
    n = int((x1 - x0) / h)
    xi = np.linspace(x0, x1, n + 1)
    yi = func(xi)
    return h / 3 * (yi[0] + yi[-1] + 4 * np.sum(yi[1:-1:2]) + 2 * np.sum(yi[2:-1:2]))

# Остаточные члены
def get_R_for_rectangle(h, x0, x1):
    return h**2 * (x1 - x0) * abs(second_derivative(0)) / 24

def get_R_for_trapezoidal(h, x0, x1):
    return h**2 * (x1 - x0) * abs(second_derivative(0)) / 12

def get_R_for_simpson(h, x0, x1):
    return h**4 * (x1 - x0) * abs(fourth_derivative(0)) / 180

# Коррекция методом Рунге-Ромберга-Ричардсона
def runge_romberg_richardson(F_h, F_kh, k, p):
    return F_h + (F_h - F_kh) / (k**p - 1)

# Основной код
h1 = 1.0
h2 = 0.5
x0 = -2.0
x1 = 2.0

# Вычисление остатков
R_rect_h1 = get_R_for_rectangle(h1, x0, x1)
R_trap_h1 = get_R_for_trapezoidal(h1, x0, x1)
R_simp_h1 = get_R_for_simpson(h1, x0, x1)

# Вычисление интегралов
F_rect_h1 = rectangle_method(x0, x1, h1)
F_rect_h2 = rectangle_method(x0, x1, h2)

F_trap_h1 = trapezoidal_method(x0, x1, h1)
F_trap_h2 = trapezoidal_method(x0, x1, h2)

F_simp_h1 = simpson_method(x0, x1, h1)
F_simp_h2 = simpson_method(x0, x1, h2)

# Коррекция методом Рунге-Ромберга-Ричардсона
RRR_rect = runge_romberg_richardson(F_rect_h1, F_rect_h2, h1 / h2, 2)
RRR_trap = runge_romberg_richardson(F_trap_h1, F_trap_h2, h1 / h2, 2)
RRR_simp = runge_romberg_richardson(F_simp_h1, F_simp_h2, h1 / h2, 4)

# Абсолютные ошибки
abs_error_rect = abs(F_rect_h2 - RRR_rect)
abs_error_trap = abs(F_trap_h2 - RRR_trap)
abs_error_simp = abs(F_simp_h2 - RRR_simp)

# Вывод результатов
print(f"Rectangle method: F(h1) = {F_rect_h1:.6f}, F(h2) = {F_rect_h2:.6f}, RRR = {RRR_rect:.6f}, Abs Error = {abs_error_rect:.6e}")
print(f"Trapezoidal method: F(h1) = {F_trap_h1:.6f}, F(h2) = {F_trap_h2:.6f}, RRR = {RRR_trap:.6f}, Abs Error = {abs_error_trap:.6e}")
print(f"Simpson method: F(h1) = {F_simp_h1:.6f}, F(h2) = {F_simp_h2:.6f}, RRR = {RRR_simp:.6f}, Abs Error = {abs_error_simp:.6e}")

# Построение графика
x = np.linspace(x0, x1, 100)
y = func(x)

plt.figure(figsize=(12, 6))
plt.plot(x, y, label="f(x) = 1 / (x^3 + 64)", color='blue')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)

plt.title("Function and Numerical Integration", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("f(x)", fontsize=12)
plt.legend()
plt.grid()
plt.show()
