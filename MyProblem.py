import numpy as np
import geatpy as ea
import random
from multiprocessing.dummy import Pool as ThreadPool
import tensorflow as tf
from sklearn import preprocessing
import pandas as pd

np.random.seed(42)
random.seed(42)


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 4  # 初始化Dim（决策变量维数）
        varTypes = [0, 0, 0, 0]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [1.75, 1, 0.2, 25]  # 决策变量下界
        ub = [4, 5, 1.2, 800]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        model = tf.keras.models.load_model('model2.h5')
        # 目标函数计算中用到的一些数据
        self.data = pd.read_csv('new_data.csv', names=['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)', 'depth of fusion(mm)', 'porosity(%)']).dropna()
        X_train = self.data.iloc[:, 0:4]
        self.std = preprocessing.StandardScaler()
        self.std.fit_transform(X_train)
        self.model = model

    def evaluation(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        pop.ObjV = np.zeros((pop.sizes, 1))

        def subAimFunc(i):
            p = Vars[i, [0]]
            v = Vars[i, [1]]
            r = Vars[i, [2]]
            f = Vars[i, [3]]
            g = pd.DataFrame([[p, v, r, f]], columns=list(['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)']))
            g = pd.DataFrame(self.std.transform(g), columns=list(['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)']))
            pop.ObjV[i] = 0.9 * self.model.predict(g)[0][1] - 0.1 * self.model.predict(g)[0][0]
        pool = ThreadPool(16)  # 设置池的大小
        pool.map(subAimFunc, list(range(pop.sizes)))

    def test(self, g):
        g = pd.DataFrame(self.std.transform(g), columns=list(['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)']))
        scores = self.model.predict(g)
        print(str(scores))
