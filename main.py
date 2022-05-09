from distMat import cityDistMat, cityPosList
from ga import Ga
from myPlot import myPlot

# 打印城市的坐标
print(cityPosList)

# 打印城市之间的距离矩阵
print(cityDistMat)

# 实例化遗传算法
ga = Ga()
# 开始训练(每一代最好的基因以及对应的适应度 )
resultList, fitnessList = ga.train()
# 把最好的当作最后的结果
result = resultList[-1]
# 最优基因的路径
resultPosList = cityPosList[result, :]

# 绘图
myPlot(resultPosList, resultList, fitnessList)