# 个体类
# 注意，在其它应用场景下，下面四个函数应重写
from fitness import my_fitness
from gen_gene import my_gen_gene


# 个体类
class Individual:
    # 依次传入基因长度、基因的适应度函数、生成基因的策略、基因（这个参数不传的话随机生成一个个体，基因是随机的）
    def __init__(self, gene=None):
        if gene is None:
            gene = my_gen_gene()
        self.gene = gene
        # 计算个体适应度
        self.fitness = my_fitness(self.gene)