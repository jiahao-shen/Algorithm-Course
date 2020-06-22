import numpy as np
from time import time
from copy import deepcopy



class Puzzle(object):

    def __init__(self, n):
        self.n = n
        self.seed()

    def seed(self):
        np.random.seed(int(time()))

    def shuffle(self, k=20):
        self.a = np.arange(self.n * self.n).reshape((self.n, self.n))
        self.location = (0, 0)

        for _ in range(k):
            action = np.random.randint(0, 4)
            self.step(action)

        return self.state()

    def step(self, action):
        MOVE = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        x, y = self.location
        new_x, new_y = x + MOVE[action, 0], y + MOVE[action, 1]

        if new_x < 0 or new_x >= self.n or new_y < 0 or new_y >= self.n:
            return self.state()

        self.a[x, y], self.a[new_x, new_y] = self.a[new_x, new_y], self.a[x, y]
        self.location = (new_x, new_y)

        return self.state()

    def render(self):
        for i in range(self.n):
            for j in range(self.n):
                print(self.a[i, j], end='\t')
            print()
        print('----------------------------')

    def state(self):
        return self.a.reshape((1, self.n * self.n))
