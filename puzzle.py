import numpy as np


class Puzzle(object):
    def __init__(self, n=3):
        self.n = n
        self.shuffle()

    def shuffle(self):
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
                        break
        else:
            self.shuffle()

    def move(self, direction):
        x, y = self.location
        if direction == 'up:'
        if x > 0:
            self.old_value = self.manhattan()
            self.a[x, y], self.a[x - 1, y] = self.a[x - 1, y], self.a[x, y]
            self.location = (x - 1, y)
            self.new_value = self.manhattan()
        elif direction == 'down':
            if x < self.n - 1:
                self.old_value = self.manhattan()
                self.a[x, y], self.a[x + 1, y] = self.a[x + 1, y], self.a[x, y]
                self.location = (x + 1, y)
                self.new_value = self.manhattan()
        elif direction == 'left':
            if y > 0:
                self.old_value = self.manhattan()
                self.a[x, y], self.a[x, y - 1] = self.a[x, y - 1], self.a[x, y]
                self.location = (x, y - 1)
                self.new_value = self.manhattan()
        elif direction == 'right':
            if y < self.n - 1:
                self.old_value = self.manhattan()
                self.a[x, y], self.a[x, y + 1] = self.a[x, y + 1], self.a[x, y]
                self.location = (x, y + 1)
                self.new_value = self.manhattan()
        self.output()

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
                    x = self.n - i // self.n
                    break
            return (x + inv_count) % 2 == 1
        else:
            return inv_count % 2 == 0

    def over(self):
        return self.new_value == 0

    def manhattan(self):
        distance = 0
        for i in range(self.n):
            for j in range(self.n):
                x = self.a[i, j] // self.n
                y = self.a[i, j] % self.n
                distance += (np.abs(x - i) + np.abs(y - j))
        return distance

    def output(self):
        print('Location:', self.location)
        print('Old Value:', self.old_value)
        print('New Value:', self.new_value)
        for i in range(self.n):
            for j in range(self.n):
                print(self.a[i, j], end='\t')
            print()
        print('----------------------------')


if __name__ == '__main__':
    p = Puzzle(3)
    print(p.validate(np.array([1, 8, 2,
                               0, 4, 3,
                               7, 6, 5])))

    p = Puzzle(4)
    print(p.validate(np.array([13, 2, 10, 3,
                               1, 12, 8, 4,
                               5, 0, 9, 6,
                               15, 14, 11, 7])))
    print(p.validate(np.array([6, 13, 7, 10,
                               8, 9, 11, 0,
                               15, 2, 12, 5,
                               14, 3, 1, 4])))
    print(p.validate(np.array([3, 9, 1, 15,
                               14, 11, 4, 6,
                               13, 0, 10, 12,
                               2, 7, 8, 5])))
