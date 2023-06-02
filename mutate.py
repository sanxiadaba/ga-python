# 变异的策略
import random
from copy import deepcopy

from constant import gene_len, mutate_prob


# 自定义基因变异的函数,这里使用的变异策略是直接翻转
def my_mutate(new_individual_list):
    for individual in new_individual_list:
        # 触发变异概率的话,基因进行翻转
        if random.random() < mutate_prob:
            # 这里采用对某个基因片段做反转的做法
            old_gene = deepcopy(individual.gene)
            index_begin = random.randint(0, gene_len - 2)
            index_end = random.randint(index_begin, gene_len - 1)
            # 要变异的片段
            gene_mutate = old_gene[index_begin:index_end]
            # 因为在python里,列表是引用类型,这里的变异,是对区间翻转
            gene_mutate.reverse()
            individual.gene = old_gene[:index_begin] + \
                gene_mutate + old_gene[index_end:]
