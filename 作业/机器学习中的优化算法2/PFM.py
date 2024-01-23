import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    """目标函数，这里以一个简单的二次函数为例"""
    return x**2 + 2*x + 1

def constraint_penalty(x, penalty_coefficient, constraint_value):
    """罚函数，用于处理约束条件"""
    return penalty_coefficient * max(0, constraint_value - x)

def total_penalty(x, penalty_coefficient, constraint_value):
    """总目标函数，包括目标函数和罚函数"""
    return target_function(x) + constraint_penalty(x, penalty_coefficient, constraint_value)

def gradient_total_penalty(x, penalty_coefficient, constraint_value):
    """总目标函数的梯度"""
    return 2*x + 2 - penalty_coefficient * (x - constraint_value > 0)

def penalty_gradient_descent(initial_x, learning_rate, num_iterations, penalty_coefficient, constraint_value):
    """罚函数法的梯度下降算法"""
    x_values = []
    y_values = []

    x = initial_x
    for _ in range(num_iterations):
        x_values.append(x)
        y_values.append(total_penalty(x, penalty_coefficient, constraint_value))

        # 更新x值
        x = x - learning_rate * gradient_total_penalty(x, penalty_coefficient, constraint_value)

    return x_values, y_values

def plot_penalty_gradient_descent(x_values, y_values):
    """绘制罚函数法的梯度下降过程"""
    plt.plot(x_values, y_values, '-o', label='Penalty Gradient Descent')
    plt.xlabel('x')
    plt.ylabel('Total Objective Function')
    plt.title('Penalty Gradient Descent Optimization')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    initial_x = 5.0  # 初始值
    learning_rate = 0.1  # 学习率
    num_iterations = 50  # 迭代次数
    penalty_coefficient = 10  # 罚项系数
    constraint_value = 2  # 约束条件的值

    x_values, y_values = penalty_gradient_descent(initial_x, learning_rate, num_iterations, penalty_coefficient, constraint_value)
    plot_penalty_gradient_descent(x_values, y_values)
