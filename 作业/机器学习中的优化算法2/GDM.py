import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    """目标函数，这里以一个简单的二次函数为例"""
    return x**2 + 2*x + 1

def gradient(x):
    """目标函数的梯度，对于二次函数，梯度为2x + 2"""
    return 2*x + 2

def gradient_descent(initial_x, learning_rate, num_iterations):
    """梯度下降法优化函数"""
    x_values = []
    y_values = []

    x = initial_x
    for _ in range(num_iterations):
        x_values.append(x)
        y_values.append(target_function(x))

        # 更新x值
        x = x - learning_rate * gradient(x)

    return x_values, y_values

def plot_gradient_descent(x_values, y_values):
    """绘制梯度下降过程"""
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

    x_values, y_values = gradient_descent(initial_x, learning_rate, num_iterations)
    plot_gradient_descent(x_values, y_values)