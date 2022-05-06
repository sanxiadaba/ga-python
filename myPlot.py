# 画图的函数
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def myPlot(resultPosList, resultList, fitnessList):
    fig = plt.figure()
    plt.plot(resultPosList[:, 0], resultPosList[:, 1], 'o-r')
    plt.title(u"路线")
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(fitnessList)
    plt.title(u"适应度曲线")
    plt.legend()
    fig.show()

    # 加上这一句，解决闪退问题
    plt.pause(0)
