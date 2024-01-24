import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)
n = 100  # 样本数量
p = 20   # 特征数量

X = np.random.randn(n, p)
true_beta = np.random.randn(p)
noise = 0.1 * np.random.randn(n)
y = np.dot(X, true_beta) + noise

# 定义LASSO目标函数和梯度
def lasso_objective(beta, X, y, lambda_):
    n = len(y)
    residuals = y - np.dot(X, beta)
    lasso_term = lambda_ * np.sum(np.abs(beta))
    objective = 0.5 * np.sum(residuals**2) + lasso_term
    return objective

def lasso_gradient(beta, X, y, lambda_):
    n = len(y)
    residuals = y - np.dot(X, beta)
    sign = np.sign(beta)
    gradient = -np.dot(X.T, residuals) + lambda_ * sign
    return gradient

# 记录目标函数值随迭代次数的变化
def callback_function(beta):
    obj_value = lasso_objective(beta, X, y, lambda_)
    objective_values.append(obj_value)

# 运行Nesterov加速算法
initial_beta = np.zeros(p)
lambda_ = 0.1
objective_values = []

def nesterov_gradient(beta, X, y, lambda_, momentum):
    n = len(y)
    residuals = y - np.dot(X, beta)
    sign = np.sign(beta)
    gradient = -np.dot(X.T, residuals) + lambda_ * sign
    return gradient + momentum * (beta - old_beta)

old_beta = initial_beta.copy()
momentum = 0.9  # 设置Nesterov加速算法的动量参数

result = minimize(
    fun=lasso_objective,
    x0=initial_beta,
    args=(X, y, lambda_),
    jac=lambda beta, X, y, lambda_: nesterov_gradient(beta, X, y, lambda_, momentum),
    method='L-BFGS-B',
    callback=callback_function
)

print(objective_values)

# 绘制目标函数值随迭代次数的变化曲线
plt.plot(objective_values[1:], marker='o')
plt.title('LASSO Objective Function Value vs Iteration (Nesterov Accelerated)')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.show()
