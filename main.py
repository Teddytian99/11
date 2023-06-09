# -*- coding: utf-8 -*-
"""main.py"""

import numpy as np
import geatpy as ea 
from MyProblem import MyProblem 
import random
np.random.seed(42)
random.seed(42)


if __name__ == '__main__':
    problem = MyProblem()  
    myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, ea.Population(Encoding='RI', NIND=40),
                                                logTras=1)  
    myAlgorithm.MAXGEN = 50  
    myAlgorithm.trappedValue = 1e-6  
    myAlgorithm.maxTrappedCount = 10  
    myAlgorithm.mutOper.F = 0.5  
    myAlgorithm.recOper.XOVR = 0.5  
    myAlgorithm.drawing = 1 

    res = ea.optimize(myAlgorithm, seed=42, verbose=False)  
    note = open('BPEA.txt', 'a+')
    note.write('best target：%s\n' % res['ObjV'][0][0])
    for i in range(res['Vars'].shape[1]):
        note.write('best variables：%s\n' % (res['Vars'][0][i]))
    note.write('generation：%s\n' % (res['nfev']))
    note.write('time %s s\n' % res['executeTime'])
    note.write(str(res['lastPop']))
    note.write("\n")
    note.write(str(res))
    note.close()

