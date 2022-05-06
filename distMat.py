# -*- coding: utf-8 -*-
# 生成、计算距离矩阵

from constant import cityNum, dimNum
import numpy as np

# 随机生成每个城市的坐标(也可以自定义)
cityPosList = np.random.rand(cityNum, dimNum)

# 城市之间两两的距离矩阵


def buildDistMat(inputList):
    n = cityNum
    distMat = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            d = inputList[i, :] - inputList[j, :]
            # 计算点积
            distMat[i, j] = np.dot(d, d)
            distMat[j, i] = distMat[i, j]
    return distMat


# 实例化距离矩阵
cityDistMat = buildDistMat(cityPosList)
