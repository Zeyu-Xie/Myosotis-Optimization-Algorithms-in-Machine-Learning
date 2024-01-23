import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    return x**2 + 2*x + 1

def gradient(x):
    return 2*x + 2

def stochastic_gradient_descent(initial_x, learning_rate, num_iterations, batch_size):
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
