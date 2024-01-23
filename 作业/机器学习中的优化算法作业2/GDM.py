import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    return x**2 + 2*x + 1

def gradient(x):
    return 2*x + 2

def gradient_descent(initial_x, learning_rate, num_iterations):
    x_values = []
    y_values = []

    x = initial_x
    for _ in range(num_iterations):
        x_values.append(x)
        y_values.append(target_function(x))

        x = x - learning_rate * gradient(x)

    return x_values, y_values

def plot_gradient_descent(x_values, y_values):
    plt.plot(x_values, y_values, '-o', label='Gradient Descent')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent Optimization')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    initial_x = 5.0  # 初始值
    learning_rate = 0.1  # 学习率
    num_iterations = 50  # 迭代次数

    x_values, y_values = gradient_descent(
        initial_x, learning_rate, num_iterations)
    plot_gradient_descent(x_values, y_values)
