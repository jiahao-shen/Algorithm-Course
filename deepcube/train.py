import gym
from dnn import *
from puzzle import *
from math import inf
from tqdm import trange
from copy import deepcopy


def over(state, n):
    goal = np.arange(n * n).reshape(1, n * n)
    return np.array_equal(state, goal)


def main():
    n = 3
    env = Puzzle(n)
    dnn = DNN(n)

    for e in trange(2000):
        for k in range(20):
            nums = 100
            train_x = np.zeros((nums, n * n))
            train_y = np.zeros((nums, 1))
            for i in range(nums):
                state = env.shuffle(k)
                y = inf
                if over(state, n):
                    y = 0
                for action in range(4):
                    new_state = deepcopy(env).step(action)
                    cost = dnn.estimate(state)
                    if over(new_state, n):
                        cost = 0
                    y = min(y, 1 + cost)

                train_x[i] = state
                train_y[i] = y

            loss = dnn.train(train_x, train_y)
            if loss < 0.1:
                dnn.update()

    dnn.save('temp.h5')


def test():
    n = 3
    dnn = DNN(n)
    dnn.load('temp.h5')

    state = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
    print(dnn.predict(state))
    state = np.array([[1, 0, 2, 3, 4, 5, 6, 7, 8]])
    print(dnn.predict(state))
    state = np.array([[1, 2, 0, 3, 4, 5, 6, 7, 8]])
    print(dnn.predict(state))
    state = np.array([[1, 4, 2, 3, 0, 5, 6, 7, 8]])
    print(dnn.predict(state))
    state = np.array([[1, 4, 2, 5, 6, 8, 3, 7, 0]])
    print(dnn.predict(state))


if __name__ == '__main__':
    # main()
    test()
