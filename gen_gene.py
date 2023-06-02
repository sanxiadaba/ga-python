# 生成基因
import random

from constant import gene_len


# 生成符合条件的随机基因的列表
def my_gen_gene():
    gene = [i for i in range(gene_len)]
    random.shuffle(gene)
    return gene
