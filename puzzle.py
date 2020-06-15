import numpy as np
from dqn import *
from time import time
from tqdm import trange


class Puzzle(object):
    def __init__(self, n=3):
        self.n = n
        self.a = np.arange(self.n * self.n).reshape((self.n, self.n))
        self.location = (0, 0)
        np.random.seed(int(time()))

    def reset(self):
        temp = np.arange(self.n * self.n)
        np.random.shuffle(temp)
        if self.validate(temp):
            self.a = temp.reshape((self.n, self.n))
            for i in range(self.n):
                for j in range(self.n):
                    if self.a[i, j] == 0:
                        self.location = (i, j)
                        self.old_value = self.manhattan()
                        self.new_value = self.old_value
                        return self.state()
        else:
            return self.reset()

    def step(self, action):
        direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        x, y = self.location
        new_x, new_y = x + direction[action][0], y + direction[action][1]

        if new_x < 0 or new_x >= self.n or new_y < 0 or new_y >= self.n:
            return self.state(), -1.0, False

        self.old_value = self.manhattan()
        self.a[x, y], self.a[new_x, new_y] = self.a[new_x, new_y], self.a[x, y]
        self.location = (new_x, new_y)
        self.new_value = self.manhattan()

        return self.state(), self.reward(), self.over()

    def validate(self, temp):
        inv_count = 0
        for i in range(self.n * self.n - 1):
            for j in range(i + 1, self.n * self.n):
                if temp[i] and temp[j] and temp[i] > temp[j]:
                    inv_count += 1

        if self.n % 2 == 0:
            x = 0
            for i in range(self.n * self.n):
                if temp[i] == 0:
                    x = i // self.n
                    break
            return (x + inv_count) % 2 == 0
        else:
            return inv_count % 2 == 0

    def manhattan(self):
        distance = 0
        for i in range(self.n):
            for j in range(self.n):
                x = self.a[i, j] // self.n
                y = self.a[i, j] % self.n
                distance += (np.abs(x - i) + np.abs(y - j))
        return distance

    def render(self):
        # print('Location:', self.location)
        # print('Old Value:', self.old_value)
        # print('New Value:', self.new_value)
        # print('Over:', self.over())
        for i in range(self.n):
            for j in range(self.n):
                print(self.a[i, j], end='\t')
            print()
        print('----------------------------')

    def over(self):
        return self.new_value == 0

    def state(self):
        return self.a.reshape((1, self.n * self.n))

    def reward(self):
        if self.over():
            return 1.0

        if self.new_value < self.old_value:
            return 0.2
        elif self.new_value == self.old_value:
            return 0
        else:
            return -0.2


def test():
    env = Puzzle(2)
    print(env.validate(np.array([0, 1, 2, 3])))
    print(env.validate(np.array([0, 1, 3, 2])))
    print(env.validate(np.array([1, 0, 2, 3])))
    print(env.validate(np.array([2, 1, 0, 3])))

    env = Puzzle(3)
    print(env.validate(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])))
    print(env.validate(np.array([1, 0, 2, 3, 4, 5, 6, 7, 8])))
    print(env.validate(np.array([0, 1, 2, 3, 4, 5, 6, 8, 7])))
    print(env.validate(np.array([3, 1, 2, 0, 4, 5, 6, 7, 8])))


def train():
    n = 2
    env = Puzzle(n)
    agent = DQN(n * n, 4)

    episode = 1000
    batch_size = 16

    for e in trange(episode):
        state = env.reset()
        score = 0
        loss = np.array([])

        for _ in range(1000):
            action = agent.act(state)
            new_state, reward, done = env.step(action)

            agent.memorize(state, action, reward, new_state, done)
            agent.replay(batch_size)

            state = new_state
            score += reward
            loss = np.append(loss, agent.loss)

            if done:
                break

        print(score, ',', np.average(loss), file=open('logs/puzzle', 'a'))

        agent.save('models/puzzle.h5')


if __name__ == '__main__':
    # test()
    train()
