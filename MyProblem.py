import numpy as np
import geatpy as ea
import random
from multiprocessing.dummy import Pool as ThreadPool
import tensorflow as tf
from sklearn import preprocessing
import pandas as pd

np.random.seed(42)
random.seed(42)


class MyProblem(ea.Problem):  
    def __init__(self):
        name = 'MyProblem' 
        M = 1  
        maxormins = [1]  
        Dim = 4  
        varTypes = [0, 0, 0, 0]  
        lb = [1.75, 1, 0.2, 25]  
        ub = [4, 5, 1.2, 800]  
        lbin = [1] * Dim 
        ubin = [1] * Dim  
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        model = tf.keras.models.load_model('model2.h5')
        self.data = pd.read_csv('new_data.csv', names=['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)', 'depth of fusion(mm)', 'porosity(%)']).dropna()
        X_train = self.data.iloc[:, 0:4]
        self.std = preprocessing.StandardScaler()
        self.std.fit_transform(X_train)
        self.model = model

    def evaluation(self, pop):  
        Vars = pop.Phen 
        pop.ObjV = np.zeros((pop.sizes, 1))

        def subAimFunc(i):
            p = Vars[i, [0]]
            v = Vars[i, [1]]
            r = Vars[i, [2]]
            f = Vars[i, [3]]
            g = pd.DataFrame([[p, v, r, f]], columns=list(['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)']))
            g = pd.DataFrame(self.std.transform(g), columns=list(['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)']))
            pop.ObjV[i] = 0.9 * self.model.predict(g)[0][1] - 0.1 * self.model.predict(g)[0][0]
        pool = ThreadPool(16)  
        pool.map(subAimFunc, list(range(pop.sizes)))

    def test(self, g):
        g = pd.DataFrame(self.std.transform(g), columns=list(['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)']))
        scores = self.model.predict(g)
        print(str(scores))
