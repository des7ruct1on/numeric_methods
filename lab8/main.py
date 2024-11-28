import numpy as np
import matplotlib.pyplot as plt

def find_index(eps, x_star, vector_x):
    """Ищет индекс x_star в векторе vector_x с заданной точностью eps."""
    for i, x in enumerate(vector_x):
        if abs(x_star - x) < eps:
            return i
    raise ValueError("x_star not found in vector_x")


def leftist_diff(eps, ind, vector_y, vector_x):
    """Вычисляет левую производную."""
    if ind == 0:
        raise ValueError("The left derivative cannot be calculated for the first element")
    dx = vector_x[ind] - vector_x[ind - 1]
    if abs(dx) < eps:
        raise ZeroDivisionError("Division by zero when calculating left diff!")
    return (vector_y[ind] - vector_y[ind - 1]) / dx


def rightist_diff(eps, ind, vector_y, vector_x):
    """Вычисляет правую производную."""
    if ind >= len(vector_x) - 1:
        raise ValueError("The right derivative cannot be calculated for the last element")
    dx = vector_x[ind + 1] - vector_x[ind]
    if abs(dx) < eps:
        raise ZeroDivisionError("Division by zero when calculating right diff!")
    return (vector_y[ind + 1] - vector_y[ind]) / dx


def centre_diff(eps, ind, x_star, vector_y, vector_x):
    """Вычисляет центральную производную."""
    if ind >= len(vector_x) - 2:
        raise ValueError("The central derivative cannot be calculated for the last two elements")
    dx1 = vector_x[ind + 1] - vector_x[ind]
    dx2 = vector_x[ind + 2] - vector_x[ind + 1]
    dx = vector_x[ind + 2] - vector_x[ind]
    if abs(dx1) < eps or abs(dx2) < eps or abs(dx) < eps:
        raise ZeroDivisionError("Division by zero when calculating center diff!")
    term1 = (vector_y[ind + 1] - vector_y[ind]) / dx1
    term2 = (vector_y[ind + 2] - vector_y[ind + 1]) / dx2
    return term1 + (term2 - term1) / dx * (2 * x_star - vector_x[ind] - vector_x[ind + 1])


def second_diff(eps, ind, x_star, vector_y, vector_x):
    """Вычисляет вторую производную."""
    if ind >= len(vector_x) - 2:
        raise ValueError("The second derivative cannot be calculated for the last two elements")
    dx1 = vector_x[ind + 1] - vector_x[ind]
    dx2 = vector_x[ind + 2] - vector_x[ind + 1]
    dx = vector_x[ind + 2] - vector_x[ind]
    if abs(dx1) < eps or abs(dx2) < eps or abs(dx) < eps:
        raise ZeroDivisionError("Division by zero when calculating second diff!")
    term1 = (vector_y[ind + 1] - vector_y[ind]) / dx1
    term2 = (vector_y[ind + 2] - vector_y[ind + 1]) / dx2
    return 2 * (term2 - term1) / dx


vector_x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
vector_y = np.array([-0.7854, 0.0, 0.7854, 1.1071, 1.249])
x_star = 1.0
eps = 1e-5

try:
    index = find_index(eps, x_star, vector_x)
    left, right, centre, second = None, None, None, None

    if index > 0:
        left = leftist_diff(eps, index, vector_y, vector_x)
    if index < len(vector_x) - 1:
        right = rightist_diff(eps, index, vector_y, vector_x)
    if index > 0 and index < len(vector_x) - 2:
        centre = centre_diff(eps, index - 1, x_star, vector_y, vector_x)
        second = second_diff(eps, index - 1, x_star, vector_y, vector_x)

    print(f"x_star = {x_star}")
    if left is not None:
        print(f"Left derivative: {left:.8f}")
    if right is not None:
        print(f"Right derivative: {right:.8f}")
    if centre is not None:
        print(f"Central derivative: {centre:.8f}")
    if second is not None:
        print(f"Second derivative: {second:.8f}")

    # График
    plt.figure(figsize=(10, 6))
    plt.plot(vector_x, vector_y, 'o-', label="Function")
    plt.axvline(x_star, color='gray', linestyle='--', label=f"x_star = {x_star}")

    if left is not None:
        plt.text(x_star - 0.5, vector_y[index], f"Left: {left:.4f}", color='red')
    if right is not None:
        plt.text(x_star + 0.5, vector_y[index], f"Right: {right:.4f}", color='green')
    if centre is not None:
        plt.text(x_star, vector_y[index] + 0.1, f"Central: {centre:.4f}", color='blue')
    if second is not None:
        plt.text(x_star, vector_y[index] - 0.2, f"Second: {second:.4f}", color='purple')

    plt.title("Function and Derivatives")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

except Exception as e:
    print(f"Error: {e}")