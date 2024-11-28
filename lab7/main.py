import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

def solve_normal_system_mnk(x, y, degree):
    n = len(x)
    m = degree + 1  # количество коэффициентов
    A = np.zeros((m, m))
    B = np.zeros(m)

    for i in range(m):
        for j in range(m):
            A[i, j] = sum(x[k] ** (i + j) for k in range(n))
        B[i] = sum((x[k] ** i) * y[k] for k in range(n))

    # Решение системы уравнений методом Гаусса
    coeffs = np.linalg.solve(A, B)
    return coeffs


def evaluate_polynomial(coeffs, x):
    return sum(c * (x ** i) for i, c in enumerate(coeffs))


def calculate_error_sum(x, y, coeffs):
    error_sum = 0
    for j in range(len(x)):
        F = evaluate_polynomial(coeffs, x[j])
        error_sum += (F - y[j]) ** 2
    return error_sum


def plot_results(x, y, coeffs_manual, coeffs_builtin, degree):
    x_fit = np.linspace(min(x), max(x), 500)
    y_fit_manual = [evaluate_polynomial(coeffs_manual, xi) for xi in x_fit]
    y_fit_builtin = np.polyval(coeffs_builtin[::-1], x_fit)

    plt.scatter(x, y, color='red', label='Data Points')
    plt.plot(x_fit, y_fit_manual, label=f'Custom Fit (degree {degree})', linestyle='--', color='blue')
    plt.plot(x_fit, y_fit_builtin, label=f'Numpy Fit (degree {degree})', linestyle='-', color='green')
    plt.title(f'Polynomial Fit of Degree {degree}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()


x = np.array([-0.7, -0.4, -0.1, 0.2, 0.5, 0.8])
y = np.array([-0.7754, -0.41152, -0.10017, 0.20136, 0.5236, 0.9273])

# Полином первой степени
degree = 1
coeffs_manual = solve_normal_system_mnk(x, y, degree)
coeffs_builtin = np.polyfit(x, y, degree)
print(f"Custom coefficients for degree {degree}: {coeffs_manual}")
print(f"Numpy coefficients for degree {degree}: {coeffs_builtin[::-1]}")  # polyfit возвращает старшие коэффициенты первыми

error_sum_manual = calculate_error_sum(x, y, coeffs_manual)
error_sum_builtin = np.sum((np.polyval(coeffs_builtin, x) - y) ** 2)
print(f"Custom error sum for degree {degree}: {error_sum_manual:.5f}")
print(f"Numpy error sum for degree {degree}: {error_sum_builtin:.5f}")
    
plot_results(x, y, coeffs_manual, coeffs_builtin, degree)

# Полином второй степени
degree = 2
coeffs_manual = solve_normal_system_mnk(x, y, degree)
coeffs_builtin = np.polyfit(x, y, degree)
print(f"\nCustom coefficients for degree {degree}: {coeffs_manual}")
print(f"Numpy coefficients for degree {degree}: {coeffs_builtin[::-1]}")

error_sum_manual = calculate_error_sum(x, y, coeffs_manual)
error_sum_builtin = np.sum((np.polyval(coeffs_builtin, x) - y) ** 2)
print(f"Custom error sum for degree {degree}: {error_sum_manual:.5f}")
print(f"Numpy error sum for degree {degree}: {error_sum_builtin:.5f}")
    
plot_results(x, y, coeffs_manual, coeffs_builtin, degree)
