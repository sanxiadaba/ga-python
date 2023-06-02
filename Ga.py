# 遗传算法
import random
from copy import deepcopy
from typing import List

from constant import group_num, group_num_rate, iteration_time, survivors_num
from cross import my_cross
from individual import Individual
from mutate import my_mutate


# 定义算法的具体流程
class Ga:

    def __init__(self, fitness_flag=False):
        self.best_list = [None]  # 每一代的最佳个体
        self.best = self.best_list[-1]  # 经历了整个算法之后的最佳个体
        self.individual_list = []  # 当代的个体列表
        self.result_list = []  # 每一代对应的解(最佳个体对应的基因)
        self.fitness_list = []  # 每一代的最佳适应度
        # fitness越大越好还是越小越好,默认false表示越小越好
        self.reverse_flag = fitness_flag

    # 基因重组
    def cross(self):
        return my_cross(self.individual_list)

    # 在执行基因重组的时候是可能变异的
    def mutate(self, new_individual_list):
        my_mutate(new_individual_list)
        # 两代合并(上一代与经过重组、变异的一代)
        self.individual_list += new_individual_list

    # 挑选优秀的基因 这里采用的是锦标赛挑选机制,分成一定个小组,每个小组选出一定个数的基因(为了避免局部最优的陷阱)
    def select(self):
        group_size = (survivors_num // group_num) * 2  # 每小组人数
        group_winner = int(group_size * group_num_rate)  # 每小组获胜人数
        winners = []  # 锦标赛结果
        for _ in range(group_num):
            group = []
            for _ in range(group_size):
                # 随机组成小组,注意,在这个机制下,有的基因可能压根因为运气差就不会被选中
                # 但这更贴合自然环境,即优秀的个体,也未免会留下后代
                player = random.choice(self.individual_list)
                player = Individual(player.gene)
                group.append(player)
            group = Ga.rank(group)
            # 取出获胜者
            if self.reverse_flag == False:
                winners += group[:group_winner]
            else:
                winners += group[group_size - group_winner:]
            # 新一代
        self.individual_list = winners

    # 根据数组里每个基因的fitness进行指定规则的排序(看reverse_flag)
    @staticmethod
    def rank(group: List[Individual]):
        # group里是一个个的Individual,它们按照fitness进行升序
        return sorted(group, key=lambda x: x.fitness)

    # 开始迭代
    def next_gen(self):
        # 交叉
        new_individual_list = self.cross()
        # 变异
        self.mutate(new_individual_list)
        # 选择
        self.select()
        # 对新的一代进行排序
        self.individual_list = self.rank(self.individual_list)
        # 排序后的个体列表,最后一个就是这一代中的“优胜者”
        if self.reverse_flag == False:
            self.best_list.append(self.individual_list[0])
        else:
            self.best_list.append(self.individual_list[-1])

        # 与历代最好进行比较
        if self.reverse_flag == False:
            if (self.best_list[-1]).fitness < self.best.fitness:
                self.best = self.best_list[-1]
        else:
            if (self.best_list[-1]).fitness > self.best.fitness:
                self.best = self.best_list[-1]

    # 训练
    def train(self):
        # 初代种群随机初始化
        self.individual_list = [Individual() for _ in range(survivors_num)]
        self.best = self.rank(self.individual_list)[-1]
        # 迭代
        for i in range(iteration_time):
            self.next_gen()
            print("第", i + 1, "轮迭代", "此时最佳个体基因为：", self.best.gene, " ",
                  "此时适应度为： ", self.best.fitness)
        # 更新result_list 和 fitness_list
        # 注意,先别忘了删除best_list的第一个 因为一开始放进去了一个none
        self.best_list.pop(0)
        for best in self.best_list:
            self.result_list.append(best)
            self.fitness_list.append(best.fitness)
