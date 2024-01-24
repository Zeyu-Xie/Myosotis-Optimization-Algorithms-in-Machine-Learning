import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)
n = 100  # 样本数量
p = 20   # 特征数量

X = np.random.randn(n, p)
true_beta = np.random.randn(p)
noise = 0.1 * np.random.randn(n)
y = np.dot(X, true_beta) + noise

# 定义LASSO目标函数
def lasso_objective(beta, X, y, lambda_, rho, z):
    n = len(y)
    residuals = y - np.dot(X, beta)
    lasso_term = lambda_ * np.sum(np.abs(z))
    augmented_term = (rho / 2) * np.sum((beta - z + u)**2)
    objective = 0.5 * np.sum(residuals**2) + lasso_term + augmented_term
    return objective

# 初始化参数
beta = np.zeros(p)
z = np.zeros(p)
u = np.zeros(p)
rho = 1.0  # 步长

# ADMM迭代
max_iterations = 100
lambda_ = 0.1
objective_values = []

for iteration in range(max_iterations):
    # 求解beta
    beta = np.linalg.solve(np.dot(X.T, X) + rho * np.identity(p), np.dot(X.T, y) + rho * (z - u))

    # 求解z（软阈值运算）
    z = np.maximum(0, beta + u - lambda_ / rho) - np.maximum(0, -beta - u - lambda_ / rho)

    # 更新u
    u = u + beta - z

    # 计算目标函数值
    obj_value = lasso_objective(beta, X, y, lambda_, rho, z)
    objective_values.append(obj_value)

# 打印最终结果
print("Optimal beta:", beta)
print("Objective values:", objective_values)

# 绘制目标函数值随迭代次数的变化曲线
plt.plot(objective_values[1:], marker='o')
plt.title('LASSO Objective Function Value vs Iteration (ADMM)')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.show()
