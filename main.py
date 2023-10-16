# This is a sample Python script.
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def gemel(theta, X):
    y_pred = X @ theta
    return y_pred


def costFunction(x, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum(np.square((x @ theta) - y))


def five(real_theta, x, y):
    grad = (np.transpose(x)) @ ((x @ real_theta) - y)
    return grad


def gradCalc(x, y, theta):
    return (1 / len(x)) * (x.T @ ((x @ theta) - y))


def gradientDescent(x, y, theta, alpha):
    m = len(y)
    iterations = 20

    y = y.reshape(m, 1)
    costhistory = np.zeros(iterations)
    for i in range(iterations):
        costhistory[i] = costFunction(x, y, theta)

        theta = theta - (alpha * (gradCalc(x, y, theta)))

    plt.plot(np.arange(iterations), costhistory)
    plt.show()
    return costhistory


def mini_batch(x, y, alpha, batch):
    k = 100
    N = len(x) / 60
    new_cost = np.zeros(k)
    theta = np.zeros((len(new_x[0]), 1))
    y = y.reshape(len(x), 1)
    for i in range(k):
        # new x and new y
        new_cost[i] = costFunction(x, y, theta)
        x = np.array(x)
        y = np.array(y)
        theta -= alpha * (gradCalc(x, y, theta))
    plt.plot(np.arange(k), new_cost)
    plt.show()
    return new_cost


def momentom(x, y, Beta, alpha, theta):
    v = np.zeros((len(x[0]), 1))
    k = 21
    cost = np.zeros(k)
    y = y.reshape((len(x), 1))
    for i in range(k):
        x = np.array(x)
        y = np.array(y)
        cost[i] = costFunction(x, y, theta)
        v = (v * Beta) + (alpha * gradCalc(x, y, theta))
        theta = theta - v
    plt.plot(np.arange(k), cost)
    plt.show()
    return cost


if __name__ == '__main__':
    data = pd.read_csv('cancer_data.csv', header=None)
    x = data.iloc[:, :-1].values
    x = np.array(x)  # convert x to a numpy array
    x_new = np.ones((len(x), len(x[0]) + 1))  # create a new numpy array with an additional column of ones
    x_new[:, 1:] = x

    y = data.iloc[:, -1].values



    v = np.mean(x, axis=0)
    Variance = np.var(x, axis=0)

    new_x = [[0 for _ in range(len(x[0]))] for _ in range(len(x))]
    for i in range(len(x[0])):
        for j in range(len(x)):
            sui = (x[j][i] - v[i]) / math.sqrt(Variance[i])
            new_x[j][i] = sui
    ones_col = np.ones((len(x), 1))
    new_x = np.hstack((ones_col, new_x))
    # print(len(new_x[0]))
    old_theta = np.zeros((len(new_x[0]), 1))
    gradientDescent(new_x,y,old_theta,0.1)
    mini_batch(new_x,y,0.01,1000)
    momentom(new_x,y,0.01,0.01,old_theta)


