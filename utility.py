# 一些用到的函数

from math import sqrt

from constant import city_pos


# 两两城市之间的距离
def city_dist_mat(pos1, pos2):
    dist = sqrt((city_pos[pos1][0] - city_pos[pos2][0])**2 +
                (city_pos[pos1][1] - city_pos[pos2][1])**2)
    return dist
