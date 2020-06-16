import gym
import numpy as np
from time import time
from gym import spaces
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import TensorBoard
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class Puzzle(gym.Env):

    def __init__(self, n=3):
        self.n = n
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=n * n - 1, shape=(n * n,), dtype=np.int8)

        self.seed()
        self.reset()

    def seed(self, **kwargs):
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
        direction = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        x, y = self.location
        new_x, new_y = x + direction[action, 0], y + direction[action, 1]

        if new_x < 0 or new_x >= self.n or new_y < 0 or new_y >= self.n:
            return self.state(), -1.0, False, {'info': 'Bound'}

        self.old_value = self.manhattan()
        self.a[x, y], self.a[new_x, new_y] = self.a[new_x, new_y], self.a[x, y]
        self.location = (new_x, new_y)
        self.new_value = self.manhattan()

        if self.over():
            return self.state(), 1.0, True, {'info': 'Success'}

        return self.state(), self.reward(), False, {'info': 'Step'}

    def render(self, mode=None):
        for i in range(self.n):
            for j in range(self.n):
                print(self.a[i, j], end='\t')
            print()
        print('----------------------------')

    def state(self):
        return self.a.flatten()

    def over(self):
        return self.new_value == 0

    def reward(self):
        return (self.old_value - self.new_value) / 10

    def validate(self, temp):
        inv_count = 0
        for i in range(self.n * self.n - 1):
            for j in range(i + 1, self.n * self.n):
                if temp[i] and temp[j] and temp[i] > temp[j]:
                    inv_count += 1

        if self.n % 2 == 0:
            for i in range(self.n * self.n):
                if temp[i] == 0:
                    x = i // self.n
                    return (x + inv_count) % 2 == 0
        else:
            return inv_count % 2 == 0

    def manhattan(self):
        dis = 0
        for i in range(self.n):
            for j in range(self.n):
                x = self.a[i, j] // self.n
                y = self.a[i, j] % self.n
                dis += (np.abs(x - i) + np.abs(y - j))

        return dis


def train_1():
    n = 2
    env = Puzzle(n)

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))

    memory = SequentialMemory(limit=10000, window_length=1)
    policy = BoltzmannQPolicy()

    agent = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory,
                     nb_steps_warmup=100, target_model_update=1e-2,
                     policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mse'])

    agent.fit(env, nb_steps=500000, nb_max_episode_steps=100, visualize=False,
              verbose=2, callbacks=[TensorBoard(log_dir='temp')])

    agent.save_weights('model/puzzle_dqn_2.h5')

    # agent.load_weights('model/puzzle_dqn_2.h5')
    agent.test(env, nb_episodes=5, nb_max_episode_steps=50, visualize=True)


def train_2():
    n = 3
    env = Puzzle(n)

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))

    memory = SequentialMemory(limit=10000, window_length=1)
    policy = BoltzmannQPolicy()

    agent = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory,
                     nb_steps_warmup=100, target_model_update=1e-2,
                     policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mse'])

    agent.fit(env, nb_steps=5000000, nb_max_episode_steps=500, visualize=False,
              verbose=2, callbacks=[TensorBoard(log_dir='temp')])

    agent.save_weights('model/puzzle_dqn_3.h5')

    # agent.load_weights('model/puzzle_dqn_3.h5')
    # agent.test(env, nb_episodes=5, nb_max_episode_steps=50, visualize=True)


def train_3():
    n = 4
    env = Puzzle(n)

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))

    memory = SequentialMemory(limit=10000, window_length=1)
    policy = BoltzmannQPolicy()

    agent = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory,
                     nb_steps_warmup=100, target_model_update=1e-2,
                     policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mse'])

    agent.fit(env, nb_steps=10000000, nb_max_episode_steps=1000,
              visualize=False, verbose=2,
              callbacks=[TensorBoard(log_dir='temp')])

    agent.save_weights('model/puzzle_dqn_3.h5')

    # agent.load_weights('model/puzzle_dqn_3.h5')
    # agent.test(env, nb_episodes=5, nb_max_episode_steps=50, visualize=True)


if __name__ == '__main__':
    # train_1()
    train_2()
    train_3()
