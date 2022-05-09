# 基因算法的定义

import random
from constant import cityNum, individualNum, mutateProb, iterationTime
from distMat import cityDistMat
from typing import List
from copy import deepcopy

# 基因长度等于城市的长度
geneLen = cityNum


# 个体类
class Individual:
    # 默认的话是随机生成序列
    def __init__(self, genes=None):
        if genes is None:
            genes = [i for i in range(geneLen)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluateFitness()

    # 计算个体适应度
    def evaluateFitness(self):
        fitness = 0
        for i in range(geneLen - 1):
            # 起始城市和目标城市
            fitness += cityDistMat[self.genes[i], self.genes[i + 1]]
        # 最后还要回来
        fitness += cityDistMat[self.genes[-1], self.genes[0]]
        return fitness


# 定义算法的具体流程


class Ga:

    def __init__(self):
        self.best = None  # 每一代的最佳个体
        self.individualList = []  # 每一代的个体列表
        self.resultList = []  # 每一代对应的解
        self.fitnessList = []  # 每一代对应的适应度

    # 基因重组
    def cross(self):
        # 新基因
        newGen = []
        # 先打乱再进行基因重组
        random.shuffle(self.individualList)
        # 这里步长为2，正好对应父母亲基因进行重组
        for i in range(0, individualNum - 1, 2):
            # 要进行交换的基因 注意这里用的是deepcopy,不然的话会改变原数组(把基因重组、变异后的目标与原样本放在一起进行筛选)
            geneFirst = deepcopy(self.individualList[i].genes)
            geneSecond = deepcopy(self.individualList[i + 1].genes)
            # 随机生成要交换基因的首尾，保证后者大于前者，进行交换
            indexBegin = random.randint(0, geneLen - 2)
            indexEnd = random.randint(indexBegin, geneLen - 1)
            # 记录下初始基因对应的位置(储存为字典)，为了下一步的操作
            dictFirst = {value: index for index, value in enumerate(geneFirst)}
            dictSecond = {
                value: index
                for index, value in enumerate(geneSecond)
            }

            # 交叉 注意，这里如果要交换的基因片段在另一部分已经有的话，相当于把另一个基因要重复的地方交换
            # e,g
            # [1,4,5,2,3] 与 [4,1,3,2,5]交换中间的三个基因
            # 直接交换的话，基因里面会出现重复元素
            # 1,3,2这是第二个基因“移植到”第一个基因上的
            # 首先在第一个基因里查找1， 发现不在交换的片段 于是1与4交换位置
            # 基因变为 [4,1,5,2,3]
            # 再查找3 发现基因一的3的index也在交换之外 那就交换基因1里5，3位置变为
            # [4,1,3,2,5]
            # 最后 2与2交换
            for j in range(indexBegin, indexEnd):
                value1, value2 = geneFirst[j], geneSecond[j]
                pos1, pos2 = dictFirst[value2], dictSecond[value1]
                geneFirst[j], geneFirst[pos1] = geneFirst[pos1], geneFirst[j]
                geneSecond[j], geneSecond[pos2] = geneSecond[pos2], geneSecond[
                    j]
                dictFirst[value1], dictFirst[value2] = pos1, j
                dictSecond[value1], dictSecond[value2] = j, pos2
            newGen.append(Individual(geneFirst))
            newGen.append(Individual(geneSecond))
        return newGen

    # 在执行基因重组的时候是可能变异的
    def mutate(self, newGen):
        for individual in newGen:
            # 触发变异概率的话,基因进行翻转
            if random.random() < mutateProb:
                # 这里采用对某个基因片段做反转的做法
                oldGenes = deepcopy(individual.genes)
                indexBegin = random.randint(0, geneLen - 2)
                indexEnd = random.randint(indexBegin, geneLen - 1)
                genesMutate = oldGenes[indexBegin:indexEnd]
                genesMutate.reverse()
                individual.genes = oldGenes[:indexBegin] + \
                    genesMutate + oldGenes[indexEnd:]
        # 两代合并(上一代与经过重组、变异的一代)
        self.individualList += newGen

    # 挑选优秀的基因 这里采用的是锦标赛挑选机制，分成一定个小组，每个小组选出一定个数的基因(为了避免局部最优的陷阱)
    def select(self):
        groupNum = 10  # 小组数
        groupSize = 10  # 每小组人数
        groupWinner = individualNum // groupNum  # 每小组获胜人数
        winners = []  # 锦标赛结果
        for i in range(groupNum):
            group = []
            for j in range(groupSize):
                # 随机组成小组
                player = random.choice(self.individualList)
                player = Individual(player.genes)
                group.append(player)
            group = Ga.rank(group)
            # 取出获胜者
            winners += group[:groupWinner]
        self.individualList = winners

    # 根据数组里每个基因的fitness进行升序排列
    @staticmethod
    def rank(group: List[IndentationError]):
        return sorted(group, key=lambda x: x.fitness)

    # 开始迭代
    def nextGen(self):
        # 交叉
        newGen = self.cross()
        # 变异
        self.mutate(newGen)
        # 选择
        self.select()

        # 获得这一代最好结果当作这一代的结果
        for individual in self.individualList:
            if individual.fitness < self.best.fitness:
                self.best = individual

    # 训练
    def train(self):
        # 初代种群随机初始化
        self.individualList = [Individual() for _ in range(individualNum)]
        self.best = self.individualList[0]
        # 迭代
        for i in range(iterationTime):
            self.nextGen()
            # 连接首尾(最后还要回来的)
            result = deepcopy(self.best.genes)
            result.append(result[0])

            self.resultList.append(result)
            self.fitnessList.append(self.best.fitness)
        # 返回的是每一代最好的基因和对应的适应度
        return self.resultList, self.fitnessList