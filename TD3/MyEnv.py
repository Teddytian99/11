import numpy as np
from gym import Env, spaces
import pandas as pd
import tensorflow as tf
from joblib import load
import random

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


class MyEnv(Env):
    """
    动作空间和状态空间都是连续的，奖励为机器学习预测的熔深与板厚之比减去孔隙率
    """

    def __init__(self):
        self.state = [[2.875, 2.25, 0.7, 212.5]]
        self.seed(42)
        self.std = load('D:/Users/12171/anaconda3/envs/tf28/Lib/site-packages/gym/envs/my_env/std.joblib')
        model = tf.keras.models.load_model('D:/Users/12171/anaconda3/envs/tf28/Lib/site-packages/gym/envs/my_env/model2.h5')
        self.model = model
        self.observation_space = spaces.Box(low=np.array([[1.75, 1, 0.2, 25]]), high=np.array([[4, 3.5, 1.2, 400]]), shape=(1, 4), dtype=np.float64)
        self.action_space = spaces.Box(low=np.array([[-1, -1, -1, -1]]), high=np.array([[1, 1, 1, 1]]), shape=(1, 4), dtype=np.float64)
        self.action = np.zeros((1, 4))

    def reset(self, **kwargs):
        self.state = [[2.875, 2.25, 0.7, 212.5]]
        return self.state

    def run(self, next_state, state):
        p = pd.DataFrame(next_state, columns=list(['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)']))
        p_t = pd.DataFrame(self.std.transform(p), columns=list(['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)']))
        p1 = pd.DataFrame(state, columns=list(['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)']))
        p_t1 = pd.DataFrame(self.std.transform(p1), columns=list(['p(Kw)', 'speed(m/min)', 'r(mm)', 'frequency(Hz)']))
        mh = self.model.predict(p_t)[0][0]
        por = self.model.predict(p_t)[0][1]
        mh1 = self.model.predict(p_t1)[0][0]
        por1 = self.model.predict(p_t1)[0][1]
        reward = -0.1 * mh + 0.9 * por
        reward1 = -0.1 * mh1 + 0.9 * por1
        return 100 * (reward1-reward)

    def step(self, action):
        a = [[0.0045, 0.005, 0.002, 0.75]]
        next_state = np.clip(self.state + np.multiply(a, action), self.observation_space.low, self.observation_space.high)  # 通过两个关键字查找状态转移表中的后继状态
        self.action = action
        reward = self.run(next_state, self.state)
        self.state = next_state
        done = False
        info = {}
        return next_state, reward, done, info

    def render(self, **kwargs):
        print(self.state)
        print(self.action)
