# 基因重组的策略
import random
from copy import deepcopy
from typing import List

from constant import gene_len, survivors_num
from individual import Individual


# 自定义交换基因的函数,因为这里解决的是旅行商问题，基因里每个数字只能出现一次，所以比较复杂
def my_cross(individual_list: List[Individual]):
    # 新基因
    new_individual_list = []
    # 先打乱再进行基因重组
    random.shuffle(individual_list)
    # 这里步长为2,正好对应父母亲基因进行重组
    for i in range(0, survivors_num - 1, 2):
        # 要进行交换的基因 注意这里用的是deepcopy,不然的话会改变原数组(把基因重组、变异后的目标与原样本放在一起进行筛选)
        gene_first = deepcopy(individual_list[i].gene)
        gene_second = deepcopy(individual_list[i + 1].gene)
        # 随机生成要交换基因的首尾,保证后者大于前者,进行交换
        index_begin = random.randint(0, gene_len - 2)
        index_end = random.randint(index_begin, gene_len - 1)
        # 记录下初始基因对应的位置(储存为字典),为了下一步的操作
        dict_first = {value: index for index, value in enumerate(gene_first)}
        dict_second = {value: index for index, value in enumerate(gene_second)}

        # 交叉 注意,这里如果要交换的基因片段在另一部分已经有的话,相当于把另一个基因要重复的地方交换
        # e,g
        # [1,4,5,2,3] 与 [4,1,3,2,5]交换中间的三个基因
        # 直接交换的话,基因里面会出现重复元素
        # 1,3,2这是第二个基因“移植到”第一个基因上的
        # 首先在第一个基因里查找1, 发现不在交换的片段 于是1与4交换位置
        # 基因变为 [4,1,5,2,3]
        # 再查找3 发现基因一的3的index也在交换之外 那就交换基因1里5,3位置变为
        # [4,1,3,2,5]
        # 最后 2与2交换
        for j in range(index_begin, index_end):
            value1, value2 = gene_first[j], gene_second[j]
            pos1, pos2 = dict_first[value2], dict_second[value1]
            gene_first[j], gene_first[pos1] = gene_first[pos1], gene_first[j]
            gene_second[j], gene_second[pos2] = gene_second[pos2], gene_second[
                j]
            dict_first[value1], dict_first[value2] = pos1, j
            dict_second[value1], dict_second[value2] = j, pos2
        new_individual_list.append(Individual(gene_first))
        new_individual_list.append(Individual(gene_second))
    return new_individual_list
