# -*- coding: utf-8 -*-
"""main.py"""

import numpy as np
import geatpy as ea  # import geatpy
from MyProblem import MyProblem  # 导入自定义问题接口
import random
np.random.seed(42)
random.seed(42)


if __name__ == '__main__':
    """===============================实例化问题对象==========================="""
    problem = MyProblem()  # 生成问题对象
    """=================================种群设置==============================="""
    """===============================算法参数设置============================="""
    myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, ea.Population(Encoding='RI', NIND=40),
                                                logTras=1)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 50  # 最大进化代数
    myAlgorithm.trappedValue = 1e-6  # “进化停滞”判断阈值
    myAlgorithm.maxTrappedCount = 10  # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
    myAlgorithm.mutOper.F = 0.5  # 设置差分进化的变异缩放因子
    myAlgorithm.recOper.XOVR = 0.5  # 设置交叉概率
    myAlgorithm.drawing = 1  # 设置绘图方式
    """==========================调用算法模板进行种群进化======================="""

    res = ea.optimize(myAlgorithm, seed=42, verbose=False)  # 执行算法模板
    # 输出结果
    note = open('BPEA进化结果.txt', 'a+')
    note.write('最优的目标函数值为：%s\n' % res['ObjV'][0][0])
    for i in range(res['Vars'].shape[1]):
        note.write('最优的控制变量值为：%s\n' % (res['Vars'][0][i]))
    note.write('进化次数：%s\n' % (res['nfev']))
    note.write('时间已过 %s 秒\n' % res['executeTime'])
    note.write(str(res['lastPop']))
    note.write("\n")
    note.write(str(res))
    note.close()
    """=================================检验结果==============================="""

