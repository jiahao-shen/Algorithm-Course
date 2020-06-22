import os
import gym
import json
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt

from gym.utils import seeding
from gym import error, spaces, utils
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import TensorBoard
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class MazeEnv(gym.Env):

    ACTION = ['U', 'D', 'R', 'L']

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True):

        self.viewer = None
        self.enable_render = enable_render

        if maze_file:
            self.maze_view = MazeView2D(maze_name='OpenAI Gym - Maze (%s)' % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640),
                                        enable_render=enable_render)
        elif maze_size:
            if mode == 'plus':
                has_loops = True
                num_portals = int(round(min(maze_size)/3))
            else:
                has_loops = False
                num_portals = 0

            self.maze_view = MazeView2D(maze_name='OpenAI Gym - Maze (%d x %d)' % maze_size,
                                        maze_size=maze_size, screen_size=(
                                            640, 640),
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render)
        else:
            raise AttributeError(
                'One must supply either a maze_file path (str) or the maze_size (tuple of length 2)')

        self.maze_size = self.maze_view.maze_size

        self.action_space = spaces.Discrete(2*len(self.maze_size))

        low = np.zeros(len(self.maze_size), dtype=int)
        high = np.array(self.maze_size, dtype=int) - \
            np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        self.state = None
        self.steps_beyond_done = None

        self.seed()
        self.reset()

        self.configure()

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.maze_view.move_robot(self.ACTION[action])

        if np.array_equal(self.maze_view.robot, self.maze_view.goal):
            reward = 1
            done = True
        else:
            reward = -0.1/(self.maze_size[0]*self.maze_size[1])
            done = False

        self.state = self.maze_view.robot

        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.maze_view.reset_robot()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode='human', close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)


class MazeView2D:

    def __init__(self, maze_name='Maze2D', maze_file_path=None,
                 maze_size=(30, 30), screen_size=(600, 600),
                 has_loops=False, num_portals=0, enable_render=True):

        pygame.init()
        pygame.display.set_caption(maze_name)
        self.clock = pygame.time.Clock()
        self.__game_over = False
        self.__enable_render = enable_render

        if maze_file_path is None:
            self.__maze = Maze(maze_size=maze_size,
                               has_loops=has_loops, num_portals=num_portals)
        else:
            if not os.path.exists(maze_file_path):
                dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_path = os.path.join(
                    dir_path, 'maze_samples', maze_file_path)
                if os.path.exists(rel_path):
                    maze_file_path = rel_path
                else:
                    raise FileExistsError('Cannot find %s.' % maze_file_path)
            self.__maze = Maze(maze_cells=Maze.load_maze(maze_file_path))

        self.maze_size = self.__maze.maze_size
        if self.__enable_render is True:
            self.screen = pygame.display.set_mode(screen_size)
            self.__screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))

        self.__entrance = np.zeros(2, dtype=int)
        self.__goal = np.array(self.maze_size) - np.array((1, 1))
        self.__robot = self.entrance

        if self.__enable_render is True:
            self.background = pygame.Surface(self.screen.get_size()).convert()
            self.background.fill((255, 255, 255))

            self.maze_layer = pygame.Surface(
                self.screen.get_size()).convert_alpha()
            self.maze_layer.fill((0, 0, 0, 0,))

            self.__draw_maze()
            self.__draw_portals()
            self.__draw_robot()
            self.__draw_entrance()
            self.__draw_goal()

    def update(self, mode='human'):
        try:
            img_output = self.__view_update(mode)
            self.__controller_update()
        except Exception as e:
            self.__game_over = True
            self.quit_game()
            raise e
        else:
            return img_output

    def quit_game(self):
        try:
            self.__game_over = True
            if self.__enable_render is True:
                pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    def move_robot(self, dir):
        if dir not in self.__maze.COMPASS.keys():
            raise ValueError('dir cannot be %s. The only valid dirs are %s.'
                             % (str(dir), str(self.__maze.COMPASS.keys())))

        if self.__maze.is_open(self.__robot, dir):

            self.__draw_robot(transparency=0)
            self.__robot += np.array(self.__maze.COMPASS[dir])

            if self.maze.is_portal(self.robot):
                self.__robot = np.array(self.maze.get_portal(
                    tuple(self.robot)).teleport(tuple(self.robot)))

            self.__draw_robot(transparency=255)

    def reset_robot(self):

        self.__draw_robot(transparency=0)
        self.__robot = np.zeros(2, dtype=int)
        self.__draw_robot(transparency=255)

    def __controller_update(self):
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()

    def __view_update(self, mode='human'):
        if not self.__game_over:

            self.__draw_entrance()
            self.__draw_goal()
            self.__draw_portals()
            self.__draw_robot()

            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.maze_layer, (0, 0))

            if mode == 'human':
                pygame.display.flip()

            return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

    def __draw_maze(self):

        if self.__enable_render is False:
            return

        line_colour = (0, 0, 0, 255)

        for y in range(self.maze.MAZE_H + 1):
            pygame.draw.line(self.maze_layer, line_colour, (0, y * self.CELL_H),
                             (self.SCREEN_W, y * self.CELL_H))

        for x in range(self.maze.MAZE_W + 1):
            pygame.draw.line(self.maze_layer, line_colour, (x * self.CELL_W, 0),
                             (x * self.CELL_W, self.SCREEN_H))

        for x in range(len(self.maze.maze_cells)):
            for y in range(len(self.maze.maze_cells[x])):
                walls_status = self.maze.get_walls_status(
                    self.maze.maze_cells[x, y])
                dirs = ''
                for dir, open in walls_status.items():
                    if open:
                        dirs += dir
                self.__cover_walls(x, y, dirs)

    def __cover_walls(self, x, y, dirs, colour=(0, 0, 255, 15)):

        if self.__enable_render is False:
            return

        dx = x * self.CELL_W
        dy = y * self.CELL_H

        if not isinstance(dirs, str):
            raise TypeError('dirs must be a str.')

        for dir in dirs:
            if dir == 'D':
                line_head = (dx + 1, dy + self.CELL_H)
                line_tail = (dx + self.CELL_W - 1, dy + self.CELL_H)
            elif dir == 'U':
                line_head = (dx + 1, dy)
                line_tail = (dx + self.CELL_W - 1, dy)
            elif dir == 'L':
                line_head = (dx, dy + 1)
                line_tail = (dx, dy + self.CELL_H - 1)
            elif dir == 'R':
                line_head = (dx + self.CELL_W, dy + 1)
                line_tail = (dx + self.CELL_W, dy + self.CELL_H - 1)
            else:
                raise ValueError('The only valid directions are (U, D, L, R).')

            pygame.draw.line(self.maze_layer, colour, line_head, line_tail)

    def __draw_robot(self, colour=(95, 172, 252), transparency=255):

        if self.__enable_render is False:
            return

        x = int(self.__robot[0] * self.CELL_W + self.CELL_W * 0.5 + 0.5)
        y = int(self.__robot[1] * self.CELL_H + self.CELL_H * 0.5 + 0.5)
        r = int(min(self.CELL_W, self.CELL_H)/5 + 0.5)

        pygame.draw.circle(self.maze_layer, colour +
                           (transparency,), (x, y), r)

    def __draw_entrance(self, colour=(213, 235, 89), transparency=235):

        self.__colour_cell(self.entrance, colour=colour,
                           transparency=transparency)

    def __draw_goal(self, colour=(250, 129, 109), transparency=235):

        self.__colour_cell(self.goal, colour=colour, transparency=transparency)

    def __draw_portals(self, transparency=160):

        if self.__enable_render is False:
            return

        colour_range = np.linspace(0, 255, len(self.maze.portals), dtype=int)
        colour_i = 0
        for portal in self.maze.portals:
            colour = ((100 - colour_range[colour_i]) %
                      255, colour_range[colour_i], 0)
            colour_i += 1
            for location in portal.locations:
                self.__colour_cell(location, colour=colour,
                                   transparency=transparency)

    def __colour_cell(self, cell, colour, transparency):

        if self.__enable_render is False:
            return

        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError(
                'cell must a be a tuple, list, or numpy array of size 2')

        x = int(cell[0] * self.CELL_W + 0.5 + 1)
        y = int(cell[1] * self.CELL_H + 0.5 + 1)
        w = int(self.CELL_W + 0.5 - 1)
        h = int(self.CELL_H + 0.5 - 1)
        pygame.draw.rect(self.maze_layer, colour +
                         (transparency,), (x, y, w, h))

    @property
    def maze(self):
        return self.__maze

    @property
    def robot(self):
        return self.__robot

    @property
    def entrance(self):
        return self.__entrance

    @property
    def goal(self):
        return self.__goal

    @property
    def game_over(self):
        return self.__game_over

    @property
    def SCREEN_SIZE(self):
        return tuple(self.__screen_size)

    @property
    def SCREEN_W(self):
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_H(self):
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.maze.MAZE_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.maze.MAZE_H)


class Maze:

    COMPASS = {
        'U': (0, -1),
        'R': (1, 0),
        'D': (0, 1),
        'L': (-1, 0)
    }

    def __init__(self, maze_cells=None, maze_size=(10, 10), has_loops=True, num_portals=0):

        self.maze_cells = maze_cells
        self.has_loops = has_loops
        self.__portals_dict = dict()
        self.__portals = []
        self.num_portals = num_portals

        if self.maze_cells is not None:
            if isinstance(self.maze_cells, (np.ndarray, np.generic)) and len(self.maze_cells.shape) == 2:
                self.maze_size = tuple(maze_cells.shape)
            else:
                raise ValueError('maze_cells must be a 2D NumPy array.')
        else:
            if not (isinstance(maze_size, (list, tuple)) and len(maze_size) == 2):
                raise ValueError('maze_size must be a tuple: (width, height).')
            self.maze_size = maze_size
            self._generate_maze()

    def save_maze(self, file_path):

        if not isinstance(file_path, str):
            raise TypeError('Invalid file_path. It must be a str.')

        if not os.path.exists(os.path.dirname(file_path)):
            raise ValueError('Cannot find the directory for %s.' % file_path)

        else:
            np.save(file_path, self.maze_cells,
                    allow_pickle=False, fix_imports=True)

    @classmethod
    def load_maze(cls, file_path):

        if not isinstance(file_path, str):
            raise TypeError('Invalid file_path. It must be a str.')

        if not os.path.exists(file_path):
            raise ValueError('Cannot find %s.' % file_path)

        else:
            return np.load(file_path, allow_pickle=False, fix_imports=True)

    def _generate_maze(self):

        self.maze_cells = np.zeros(self.maze_size, dtype=int)

        current_cell = (random.randint(0, self.MAZE_W-1),
                        random.randint(0, self.MAZE_H-1))
        num_cells_visited = 1
        cell_stack = [current_cell]

        while cell_stack:
            current_cell = cell_stack.pop()
            x0, y0 = current_cell

            neighbours = dict()
            for dir_key, dir_val in self.COMPASS.items():
                x1 = x0 + dir_val[0]
                y1 = y0 + dir_val[1]
                if 0 <= x1 < self.MAZE_W and 0 <= y1 < self.MAZE_H:
                    if self.all_walls_intact(self.maze_cells[x1, y1]):
                        neighbours[dir_key] = (x1, y1)

            if neighbours:
                dir = random.choice(tuple(neighbours.keys()))
                x1, y1 = neighbours[dir]

                self.maze_cells[x1, y1] = self.__break_walls(
                    self.maze_cells[x1, y1], self.__get_opposite_wall(dir))

                cell_stack.append(current_cell)

                cell_stack.append((x1, y1))

                num_cells_visited += 1

        if self.has_loops:
            self.__break_random_walls(0.2)

        if self.num_portals > 0:
            self.__set_random_portals(
                num_portal_sets=self.num_portals, set_size=2)

    def __break_random_walls(self, percent):
        num_cells = int(round(self.MAZE_H*self.MAZE_W*percent))
        cell_ids = random.sample(range(self.MAZE_W*self.MAZE_H), num_cells)

        for cell_id in cell_ids:
            x = cell_id % self.MAZE_H
            y = int(cell_id/self.MAZE_H)

            dirs = random.sample(list(self.COMPASS.keys()), len(self.COMPASS))
            for dir in dirs:
                if self.is_breakable((x, y), dir):
                    self.maze_cells[x, y] = self.__break_walls(
                        self.maze_cells[x, y], dir)
                    break

    def __set_random_portals(self, num_portal_sets, set_size=2):
        num_portal_sets = int(num_portal_sets)
        set_size = int(set_size)

        max_portal_sets = int(self.MAZE_W * self.MAZE_H / set_size)
        num_portal_sets = min(max_portal_sets, num_portal_sets)

        cell_ids = random.sample(
            range(1, self.MAZE_W * self.MAZE_H - 1), num_portal_sets*set_size)

        for i in range(num_portal_sets):
            portal_cell_ids = random.sample(cell_ids, set_size)
            portal_locations = []
            for portal_cell_id in portal_cell_ids:
                cell_ids.pop(cell_ids.index(portal_cell_id))
                x = portal_cell_id % self.MAZE_H
                y = int(portal_cell_id / self.MAZE_H)
                portal_locations.append((x, y))
            portal = Portal(*portal_locations)
            self.__portals.append(portal)

            for portal_location in portal_locations:
                self.__portals_dict[portal_location] = portal

    def is_open(self, cell_id, dir):
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        if self.is_within_bound(x1, y1):
            this_wall = bool(self.get_walls_status(
                self.maze_cells[cell_id[0], cell_id[1]])[dir])
            other_wall = bool(self.get_walls_status(self.maze_cells[x1, y1])[
                              self.__get_opposite_wall(dir)])
            return this_wall or other_wall
        return False

    def is_breakable(self, cell_id, dir):
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        return not self.is_open(cell_id, dir) and self.is_within_bound(x1, y1)

    def is_within_bound(self, x, y):
        return 0 <= x < self.MAZE_W and 0 <= y < self.MAZE_H

    def is_portal(self, cell):
        return tuple(cell) in self.__portals_dict

    @property
    def portals(self):
        return tuple(self.__portals)

    def get_portal(self, cell):
        if cell in self.__portals_dict:
            return self.__portals_dict[cell]
        return None

    @property
    def MAZE_W(self):
        return int(self.maze_size[0])

    @property
    def MAZE_H(self):
        return int(self.maze_size[1])

    @classmethod
    def get_walls_status(cls, cell):
        walls = {
            'U': (cell & 0x1) >> 0,
            'R': (cell & 0x2) >> 1,
            'D': (cell & 0x4) >> 2,
            'L': (cell & 0x8) >> 3,
        }
        return walls

    @classmethod
    def all_walls_intact(cls, cell):
        return cell & 0xF == 0

    @classmethod
    def num_walls_broken(cls, cell):
        walls = cls.get_walls_status(cell)
        num_broken = 0
        for wall_broken in walls.values():
            num_broken += wall_broken
        return num_broken

    @classmethod
    def __break_walls(cls, cell, dirs):
        if 'U' in dirs:
            cell |= 0x1
        if 'R' in dirs:
            cell |= 0x2
        if 'D' in dirs:
            cell |= 0x4
        if 'L' in dirs:
            cell |= 0x8
        return cell

    @classmethod
    def __get_opposite_wall(cls, dirs):

        if not isinstance(dirs, str):
            raise TypeError('dirs must be a str.')

        opposite_dirs = ''

        for dir in dirs:
            if dir == 'U':
                opposite_dir = 'D'
            elif dir == 'D':
                opposite_dir = 'U'
            elif dir == 'R':
                opposite_dir = 'L'
            elif dir == 'L':
                opposite_dir = 'R'
            else:
                raise ValueError('The only valid directions are (U, D, L, R).')

            opposite_dirs += opposite_dir

        return opposite_dirs


class Portal:

    def __init__(self, *locations):

        self.__locations = []
        for location in locations:
            if isinstance(location, (tuple, list)):
                self.__locations.append(tuple(location))
            else:
                raise ValueError('location must be a list or a tuple.')

    def teleport(self, cell):
        if cell in self.locations:
            return self.locations[(self.locations.index(cell) + 1) % len(self.locations)]
        return cell

    def get_index(self, cell):
        return self.locations.index(cell)

    @property
    def locations(self):
        return self.__locations


def train_1():
    env = MazeEnv(maze_file='data/maze_5x5.npy')

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))

    memory = SequentialMemory(limit=10000, window_length=1)
    policy = BoltzmannQPolicy()

    agent = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=100,
                     enable_double_dqn=False, enable_dueling_network=False, policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mse'])

    agent.fit(env, nb_steps=100000, nb_max_episode_steps=200,
              visualize=True, verbose=2, callbacks=[TensorBoard(log_dir='temp')])

    agent.save_weights('model/maze_5x5_1.h5')


def train_2():
    env = MazeEnv(maze_file='data/maze_5x5.npy')

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))

    memory = SequentialMemory(limit=10000, window_length=1)
    policy = BoltzmannQPolicy()

    agent = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=100,
                     target_model_update=1e-2, enable_double_dqn=True, enable_dueling_network=False, policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mse'])

    agent.fit(env, nb_steps=100000, nb_max_episode_steps=200,
              visualize=True, verbose=2, callbacks=[TensorBoard(log_dir='temp')])

    agent.save_weights('model/maze_5x5_2.h5')


def train_3():
    env = MazeEnv(maze_file='data/maze_5x5.npy')

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))

    memory = SequentialMemory(limit=10000, window_length=1)
    policy = BoltzmannQPolicy()

    agent = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=100,
                     target_model_update=1e-2, enable_double_dqn=True, enable_dueling_network=True, policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mse'])

    agent.fit(env, nb_steps=100000, nb_max_episode_steps=200,
              visualize=True, verbose=2, callbacks=[TensorBoard(log_dir='temp')])

    agent.save_weights('model/maze_5x5_3.h5')


def draw():
    plt.title('5 x 5 Maze')

    data = np.array(json.load(open('data/maze_5x5_1.json', 'r')))
    x, y = smooth(data)
    plt.plot(x, y, color='#FA816D', label='Nature DQN')

    data = np.array(json.load(open('data/maze_5x5_2.json', 'r')))
    x, y = smooth(data)
    plt.plot(x, y, color='#D5EB59', label='Double DQN')
    
    data = np.array(json.load(open('data/maze_5x5_3.json', 'r')))
    x, y = smooth(data)
    plt.plot(x, y, color='#5FACFC', label='Dueling DQN')

    plt.legend(loc='lower right')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


def smooth(data, weight=0.99):
    x = data[:, 1]
    y = data[:, 2]
    last = y[0]
    for i in range(len(y)):
        value = last * weight + (1 - weight) * y[i]
        y[i] = value
        last = value

    return x, y


if __name__ == '__main__':
    train_1()
    train_2()
    train_3()
    draw()
