import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime

def objective_function(x):
    return x[0]**2 + x[1]**2 - 2*x[0] + 1

def constraint1(x):
    return x[0] + x[1] - 1

def constraint2(x):
    return 2*x[0] - x[1] - 3

def penalty_function(x, constraints, penalty_param):
    penalty = 0
    for constraint in constraints:
        penalty += max(0, constraint(x))**2
    return penalty_param * penalty

def gradient_descent(initial_x, learning_rate, iterations, penalty_param):
    x = np.array(initial_x, dtype=float)
    history = [x.copy()]

    for _ in range(iterations):
        gradient = approx_fprime(x, objective_function, epsilon=1e-8)

        constraint_gradients = []
        for constraint in constraints:
            constraint_gradients.append(
                approx_fprime(x, constraint, epsilon=1e-8))

        total_gradient = gradient + 2 * penalty_param * \
            np.sum([max(0, constraint(x)) * np.array(gradient)
                   for constraint in constraints], axis=0)
        x = x - learning_rate * total_gradient

        history.append(x.copy())

    return np.array(history)

# 初始点
initial_point = [-0.5, 0.5]

# 学习率
learning_rate = 0.1

# 迭代次数
iterations = 50

# 罚参数
penalty_param = 10.0

# 约束条件
constraints = [constraint1, constraint2]

# 使用梯度下降法求解带有罚函数的优化问题
history = gradient_descent(
    initial_point, learning_rate, iterations, penalty_param)

# 绘制梯度下降过程的三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_vals = np.linspace(-1, 1.5, 100)
y_vals = np.linspace(-1, 1.5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 + Y**2 - 2*X + 1

ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')
ax.scatter(history[:, 0], history[:, 1], objective_function(
    history.T), c='red', marker='o', label='Gradient Descent')
ax.set_title('Gradient Descent with Penalty Function (3D)')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('Objective Function')
ax.legend()

plt.show()
