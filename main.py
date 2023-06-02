from ga import Ga
from plot import Plot

# 实例化遗传算法
ga = Ga()
# 开始训练
ga.train()
# 整个进化历程中最优秀的个体
result = ga.best
# 输出最终最优秀的个体，以及它的适应度
print("最好的基因为：", end='')
print(result.gene)
print("最好的基因的适应度为：", end='')
print(result.fitness)
# 历代最优秀的个体以及它们的适应度(可以酌情使用)
best_list = ga.best_list
fitness_list = ga.fitness_list

# 绘图
my_plot = Plot(ga)
my_plot.plot_all()
