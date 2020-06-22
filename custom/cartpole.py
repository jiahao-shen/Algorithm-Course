import gym
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


class DQN(object):

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Dense(units=16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def memorize(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, new_state, done in minibatch:
            target = self.model.predict(state)
            target[0][action] = reward

            if not done:
                target[0][action] += self.gamma * \
                    np.amax(self.model.predict(new_state)[0])

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, file):
        self.model.load_weights(file)

    def save(self, file):
        self.model.save_weights(file)


def train():
    env = gym.make('CartPole-v0')
    agent = DQN(4, 2)

    episode = 1000
    batch_size = 32

    for e in trange(episode):
        state = env.reset().reshape((1, 4))
        score = 0

        while True:
            # env.render()
            action = agent.act(state)
            new_state, reward, done, info = env.step(action)
            new_state = np.reshape(new_state, (1, 4))

            agent.memorize(state, action, reward, new_state, done)
            agent.replay(batch_size)

            state = new_state
            score += reward

            if done:
                break

        print(score, file=open('data/cartpole.log', 'a'))
        agent.save('model/cartpole.h5')

    env.close()


def draw():
    plt.figure(figsize=(8, 4))
    plt.title('CartPole')

    y = []
    for line in open('data/cartpole.log'):
        y.append(float(line))

    x = list(range(len(y)))
    y = smooth(y)

    plt.plot(x, y, color='#FA816D')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


def smooth(data, weight=0.99):
    last = data[0]
    for i in range(len(data)):
        value = last * weight + (1 - weight) * data[i]
        data[i] = value
        last = value

    return data


if __name__ == '__main__':
    train()
    # draw()
