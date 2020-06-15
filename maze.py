import numpy as np
from dqn import *
from time import time
from tqdm import trange


class Maze:
    def __init__(self):
        self.n = 5
        self.reset()
        np.random.seed(int(time()))

    def reset(self):
        self.location = (0, 0)
        self.a = np.array([[-1, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0],
                           [1, 1, 0, 1, 0],
                           [0, 0, 1, 0, 0]])
        return self.state()

    def step(self, action):
        direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        x, y = self.location
        new_x, new_y = x + direction[action][0], y + direction[action][1]

        if new_x < 0 or new_x >= self.n or new_y < 0 or new_y >= self.n:
            return self.state(), -0.8, False

        if self.a[new_x, new_y] == 1:
            return self.state(), -0.75, False

        self.location = (new_x, new_y)
        self.a[x, y], self.a[new_x, new_y] = self.a[new_x, new_y], self.a[x, y]

        if self.over():
            return self.state(), 1.0, True
        else:
            return self.state(), -0.04, False

    def render(self):
        for i in range(self.n):
            for j in range(self.n):
                print(self.a[i, j], end='\t')
            print()
        print('----------------------------')

    def over(self):
        x, y = self.location
        return x == self.n - 1 and y == self.n - 1

    def state(self):
        return self.a.reshape((1, self.n * self.n))


def train():
    n = 5
    env = Maze()
    agent = DQN(n * n, 4)

    episode = 1000
    batch_size = 16

    for e in trange(episode):
        state = env.reset()
        score = 0
        loss = np.array([0])

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

        print(score, ',', np.average(loss), file=open('logs/maze', 'a'))

        agent.save('models/maze.h5')


def main():
    n = 5
    env = Maze()

    agent = DQN(n * n, 4)
    # agent.load('maze.h5')

    for _ in range(100):
        env.render()

        state = env.state()
        action = np.argmax(agent.model.predict(state)[0])
        action = int(input('Action:'))

        _, _, done = env.step(action)

        if done:
            break


if __name__ == '__main__':
    train()
    # main()
