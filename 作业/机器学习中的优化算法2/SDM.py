import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    """目标函数，这里以一个简单的二次函数为例"""
    return x**2 + 2*x + 1

def gradient(x):
    """目标函数的梯度，对于二次函数，梯度为2x + 2"""
    return 2*x + 2

def stochastic_gradient_descent(initial_x, learning_rate, num_iterations, batch_size):
    """随机梯度下降算法"""
    x_values = []
    y_values = []

    x = initial_x
    data_points = np.linspace(-10, 10, 100)  # 生成一些数据点用于随机选择

    for _ in range(num_iterations):
        x_values.append(x)
        y_values.append(target_function(x))

        # 随机选择小批次数据点
        batch_indices = np.random.choice(len(data_points), batch_size, replace=False)
        batch_data = data_points[batch_indices]

        # 计算小批次的平均梯度
        avg_gradient = np.mean(gradient(batch_data))

        # 更新x值
        x = x - learning_rate * avg_gradient

    return x_values, y_values

def plot_stochastic_gradient_descent(x_values, y_values):
    """绘制随机梯度下降过程"""
    plt.plot(x_values, y_values, '-o', label='Stochastic Gradient Descent')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Stochastic Gradient Descent Optimization')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    initial_x = 5.0  # 初始值
    learning_rate = 0.1  # 学习率
    num_iterations = 50  # 迭代次数
    batch_size = 5  # 小批次大小

    x_values, y_values = stochastic_gradient_descent(initial_x, learning_rate, num_iterations, batch_size)
    plot_stochastic_gradient_descent(x_values, y_values)
