import numpy as np

def g_prime(x_i, gamma):
    # Capped l1函数的导数
    return np.where((-gamma < x_i) & (x_i < gamma) & (x_i != 0), x_i / np.abs(x_i), 0)

def prox_operator(x, gamma, mu):
    # Capped l1函数的prox运算符
    return np.sign(x) * np.maximum(np.abs(x) - gamma * mu, 0)

def solve_nonconvex_relaxation(A, b, mu, gamma, alpha, max_iter):
    n = A.shape[1]
    x = np.zeros((n, 1))
    lambda_vec = np.zeros((n, 1))

    for _ in range(max_iter):
        # 更新 x
        grad_x = A.T @ (A @ x - b) + mu * np.array([g_prime(xi, gamma) for xi in x]) + lambda_vec
        x = prox_operator(x - gamma * grad_x, gamma, mu)

        # 更新拉格朗日乘子
        lambda_vec = lambda_vec + alpha * (A @ x - b)

    return x

# 示例用法
A = np.array([[1, 2], [3, 4]])  # 替换为实际的 A 矩阵
b = np.array([5, 6])  # 替换为实际的 b 向量
mu = 0.1  # 替换为实际的 mu 值
gamma = 0.01  # 替换为实际的步长参数值
alpha = 0.1  # 替换为实际的拉格朗日乘子更新步长参数值
max_iter = 100  # 替换为实际的最大迭代次数

result = solve_nonconvex_relaxation(A, b, mu, gamma, alpha, max_iter)
print("Optimal solution x:", result)
