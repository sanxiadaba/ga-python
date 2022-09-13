# 画图的函数
import matplotlib.pyplot as plt

from constant import city_pos
from Ga import Ga

# 解决中文显示问题(windows上)
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class Plot:

    def __init__(self, ga: Ga) -> None:
        self.ga = ga

    def plot_all(self):
        self.plot_tsp()
        self.plot_fitness()
        # 加上这一句,解决闪退问题
        plt.pause(0)

    def plot_fitness(self):
        fig = plt.figure()
        plt.plot(self.ga.fitness_list)
        plt.title(u"适应度曲线")
        plt.legend()
        plt.savefig("./imgs/fitness.jpg")
        fig.show()

    # 画旅行商的路线图
    def plot_tsp(self):
        fig = plt.figure()
        best = self.ga.best
        # 别忘了添加起点
        x = [city_pos[i][0] for i in best.gene]
        x.append(city_pos[best.gene[0]][0])
        y = [city_pos[i][1] for i in best.gene]
        y.append(city_pos[best.gene[0]][1])
        plt.plot(x, y, 'o-r')
        plt.title(u"路线")
        plt.legend()
        plt.savefig("./imgs/plot_tsp.jpg")
        fig.show()
