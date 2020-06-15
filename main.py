from dqn import *
from puzzle import *
from tqdm import trange


def train():
    n = 2
    env = Puzzle(n)
    agent = DQN(n * n, 4)

    episode = 100
    batch_size = 16

    for e in range(episode):
        env.reset()
        state = env.state()
        rewards = 0
        # loss = np.array([])
        # env.render()

        for _ in range(100):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            rewards += reward

            if done:
                break

            if len(agent.memory) > batch_size:
                # loss = np.append(loss, np.array(agent.replay(batch_size)))
                agent.replay(batch_size)

        # print('Episode:', e, '/', episode, ', value:', agent.epsilon,
        #       ', reward:', rewards, ', loss:', np.average(loss))
        print('Episode:', e, '/', episode, ', value:',
              agent.epsilon, ', reward:', rewards)

    agent.save('model.h5')


def main():
    n = 2
    env = Puzzle(n)
    env.reset()

    agent = DQN(n * n, 4)
    agent.load('model.h5')

    for _ in range(50):
        state = env.state()
        action = np.argmax(agent.model.predict(state)[0])
        # print('action:', action)
        _, _, done = env.step(action)
        env.render()

        if done:
            print('done')
            break


if __name__ == '__main__':
    # train()
    main()
