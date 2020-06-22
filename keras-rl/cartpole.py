import gym
import numpy as np
from time import time
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Flatten
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def train():
    env = gym.make('CartPole-v0')
    np.random.seed(int(time()))
    env.seed(int(time()))

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))
    print(model.summary())

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, enable_double_dqn=False,
                   enable_dueling_network=False, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mse'])

    dqn.fit(env, nb_steps=50000, visualize=True, verbose=2,
            callbacks=[TensorBoard(log_dir='temp')])
    dqn.save_weights('model/cartpole.h5', overwrite=True)


if __name__ == '__main__':
    train()
