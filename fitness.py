from constant import gene_len
from utility import city_dist_mat


# 自定义适应度函数
def my_fitness(gene):
    fitness = 0
    for i in range(gene_len - 1):
        # 起始城市和目标城市
        fitness += city_dist_mat(gene[i], gene[i + 1])
    # 最后还要回来(转一圈)
    fitness += city_dist_mat(gene[-1], gene[0])
    return fitness
