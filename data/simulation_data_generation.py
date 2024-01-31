import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Define constants
T = 24 * 30 * 12  # Total time steps
S = 200  # Number of spatial locations

# Define pi constant
pi = 3.141593

# Define high and low sinusoidal functions
sin_high = (np.sin(np.array(range(T)) * 2 * pi / 24) * 0.5 + 1)
cos_low = (np.cos(np.array(range(T)) * 2 * pi / (24 * 30)) * 0.5 + 1)

# Define ARIMA(2,2) function
def ARIMA22(phi1, phi2, theta1, theta2, sigma, n):
    w = np.random.normal(size=n + 100 + 1)
    x = np.random.normal(size=2)
    x = list(x)
    for i in range(2, n + 100):
        x.append(phi1 * x[-1] + phi2 * x[-2] + theta1 * w[i - 1] + theta2 * w[i])
    x = np.array(x)
    x = x[100:]
    x = x.cumsum()
    x = (x - np.mean(x)) / (np.sqrt(np.var(x)))
    return x

# Define ARIMA(2,2) with low frequency component function
def ARIMA22_low(phi1, phi2, theta1, theta2, sigma, n, span=24 * 30):
    n_span = int(n / span)
    w = np.random.normal(size=n_span + 100 + 1)
    x = np.random.normal(size=2)
    x = list(x)
    for i in range(2, n_span + 100):
        x.append(phi1 * x[-1] + phi2 * x[-2] + theta1 * w[i - 1] + theta2 * w[i])
    x = np.array(x)
    x = x[100:]
    x = x.cumsum()
    x = (x - np.mean(x)) / (np.sqrt(np.var(x)))
    x_ = np.ones(n)
    for i in range(n_span):
        x_[i * span:(i + 1) * span] = x_[i * span:(i + 1) * span] * x[i]
    return x_

# Define exponential covariance matrix function
def exponential_covariance_matrix(S, theta):
    distance_matrix = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            distance_matrix[i, j] = np.linalg.norm(i - j)  # Euclidean distance
    return np.exp(-theta * distance_matrix)

# Define Mackey-Glass time series function
def Mackey_Glass(x, h, a, b, c, tao, T):
    for t in range(tao, tao + T):
        x[t + 1] = x[t] + h * (-b * x[t] + a * x[t - tao] / (1 + x[t - tao] ** c))
    return x

# Define Mackey-Glass series generation function
def mg_series(T, a=0.2, b=0.14, c=10, tao=25, h=1):
    x = np.arange(0, T + tao + 1 + 1000, dtype=np.float64)
    x = np.cos(x * 2 * 3.14159265 / 72) + np.random.normal(size=len(x)) * 0.5
    x = Mackey_Glass(x, h, a, b, c, tao, T)
    x = x[1000:]
    x = (x - np.mean(x)) / np.sqrt(np.var(x))
    return x

# Define function to simulate data
def simulate_data(T, S, w=[1, 1, 1, 1]):
    # Unpack the parameter set
    sigma_epsilon_squared = 1
    mu_0 = 1
    Sigma_0 = 1
    theta = 1e-4
    sigma_omega_squared = 1e-4
    G = 0.8

    # Generate X(s, t), a d-dimensional spatiotemporal field of known covariates
    X = []
    local_low = []
    local_high = []
    t = np.random.normal(1, 1, S)
    p = np.random.normal(0, 1, S)
    global_low = ARIMA22_low(0.5, 0.3, 0.5, -0.5, 1, T, span=24 * 30)
    for i in range(S):
        local_low_ = ARIMA22_low(0.5, 0.3, 0.5, -0.5, 1, T, span=24 * 30)
        local_high_ = (np.sin(np.array(range(T)) * 2 * pi / (24 * t[i]) + 2 * pi * p[i]) * 0.5 + 1)
        local_low.append(local_low_)
        local_high.append(local_high_)
        X.append(w[0] * local_high_ + w[1] * local_low_ + w[2] * sin_high + w[3] * global_low)
    X = np.array(X)
    local_low = np.array(local_low)
    local_high = np.array(local_high)

    # Generate y_t, a p-dimensional vector with Markovian temporal dynamics
    y = np.zeros(T)
    y[0] = np.random.normal(mu_0, Sigma_0)  # Initial value y_0
    for t in range(1, T):
        y[t] = G * y[t - 1] + np.random.normal(0, Sigma_0)  # Eq. (4)
    y = (y - np.mean(y)) / (np.sqrt(np.var(y)))

    # Generate u(s, t), the underlying 'true' local pollution level
    omega = np.zeros((S, T))
    for t in range(T):
        omega[:, t] = np.random.multivariate_normal(np.zeros(S),
                                                     sigma_omega_squared * exponential_covariance_matrix(S, theta))
    u = X + y + omega

    # Generate z(s, t), observed pollution level with measurement error
    epsilon = np.random.randn(S, T) * np.sqrt(sigma_epsilon_squared)
    z = u + epsilon

    return local_high, local_low, sin_high, global_low, z

# Define function to simulate data with seed list
def simulate_data2(T, S, w, seed_list):
    # Unpack the parameter set
    sigma_epsilon_squared = 1
    mu_0 = 1
    Sigma_0 = 1
    theta = 1e-4
    sigma_omega_squared = 1e-4
    G = 0.8
    # Generate X(s, t), a d-dimensional spatiotemporal field of known covariates
    X = []
    local_low = []
    local_high = []
    global_high = ARIMA22(0.1, 0.2, 0.3, -0.2, 1, T)
    global_low = ARIMA22_low(0.5, 0.3, 0.5, -0.5, 1, T, span = 24*30)
    for i in range(S):
        print(i)
        np.random.seed(seed_list[i])
        local_low_ = ARIMA22_low(0.3, 0.2, 0.5, -0.2, 1, T)
        local_high_ = ARIMA22(0.1, 0.2, 0.3, -0.2, 1, T)
        local_low.append(local_low_)
        local_high.append(local_high_)
        X.append(w[0] * local_high_ + w[1] * local_low_ + w[2] * global_high + w[3] * cos_low)
    X = np.array(X)
    local_low = np.array(local_low)
    local_high = np.array(local_high)
    # Generate y_t, a p-dimensional vector with Markovian temporal dynamics
    y = np.zeros(T)
    y[0] = np.random.normal(mu_0, Sigma_0)  # Initial value y_0
    for t in range(1, T):
        print("t = ", t)
        y[t] = G * y[t-1] + np.random.normal(0, Sigma_0)  # Eq. (4)
    y = (y-np.mean(y))/(np.sqrt(np.var(y)))
    # Generate u(s, t), the underlying 'true' local pollution level
    #K = np.random.randn(S, D)  # Replace with your own method of generating K(s)
    omega = np.zeros((S, T))
    print("get y")
    for t in range(T):
        print("t = ", t) 
        omega[:, t] = np.random.multivariate_normal(np.zeros(S), sigma_omega_squared * exponential_covariance_matrix(S, theta))  # Eq. (3)
    u = X + y + omega
    
    # Generate z(s, t), observed pollution level with measurement error
    epsilon = np.random.randn(S, T) * np.sqrt(sigma_epsilon_squared)  # Eq. (1)
    z = u + epsilon
    
    return local_high, local_low, global_high, global_low, z

def simulate_data3(T, S, w, seed_list):
    # Unpack the parameter set
    sigma_epsilon_squared = 1
    mu_0 = 1
    Sigma_0 = 1
    theta = 1e-4
    sigma_omega_squared = 1e-4
    G = 0.8
    # Generate X(s, t), a d-dimensional spatiotemporal field of known covariates
    X = []
    local_low = []
    local_high = []
    global_high = mg_series(T + 2000)[:T]
    global_low = ARIMA22_low(0.5, 0.3, 0.5, -0.5, 1, T, span = 24*30)
    for i in range(S):
        np.random.seed(seed_list[i])
        #print(i)
        local_low_ = ARIMA22_low(0.3, 0.2, 0.5, -0.2, 1, T)
        local_high_ = mg_series(T + 2000)[:T]
        local_low.append(local_low_)
        local_high.append(local_high_)
        X.append(w[0] * local_high_ + w[1] * local_low_ + w[2] * global_high + w[3] * global_low)
    X = np.array(X)
    local_low = np.array(local_low)
    local_high = np.array(local_high)
    # Generate y_t, a p-dimensional vector with Markovian temporal dynamics
    y = np.zeros(T)
    y[0] = np.random.normal(mu_0, Sigma_0)  # Initial value y_0
    for t in range(1, T):
        #print("t = ", t)
        y[t] = G * y[t-1] + np.random.normal(0, Sigma_0)  # Eq. (4)
    y = (y-np.mean(y))/(np.sqrt(np.var(y)))
    # Generate u(s, t), the underlying 'true' local pollution level
    #K = np.random.randn(S, D)  # Replace with your own method of generating K(s)
    omega = np.zeros((S, T))
    #print("get y")
    for t in range(T):
        #print("t = ", t) 
        omega[:, t] = np.random.multivariate_normal(np.zeros(S), sigma_omega_squared * exponential_covariance_matrix(S, theta))  # Eq. (3)
    u = X + y + omega
    
    # Generate z(s, t), observed pollution level with measurement error
    epsilon = np.random.randn(S, T) * np.sqrt(sigma_epsilon_squared)  # Eq. (1)
    z = u + epsilon
    
    return local_high, local_low, global_high, global_low, z

np.random.seed(2023)
seed_list = np.random.randint(low = 0, high = 10000, size = 200)
index_ = range(150, 150 + 13)
w_ = [[1, 1, 1, 1], [10, 1, 1, 1], [1, 10, 1, 1], [1, 1, 10, 1], [1, 1, 1, 10], [5, 1, 1, 1], [1, 5, 1, 1], [1, 1, 5, 1], [1, 1, 1, 5], [3, 1, 1, 1], [1, 3, 1, 1], [1, 1, 3, 1], [1, 1, 1, 3]]
n_ = [50, 100, 200] 

for j in range(3):
    for i in range(13):
        S = n_[j]
        index = index_[i] + j * 13
        print(index)
        w = w_[i]
        local_high, local_low, sin_high, cos_low, z = simulate_data3( T = T, S = S, w = w, seed_list = seed_list[:S])
        df = pd.DataFrame(z.T)
