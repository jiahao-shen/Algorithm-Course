class Node(object):
    MOVE = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    def __init__(self):
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.tiles.tostring())
        
        return self.hash
    
    def __eq__(self, other):
        return np.array_equal(self.tiles, other.tiles)

    def reset(self):

    
    def render(self):
        for i in range(m):
            for j in range(n):
                print(self.puzzle[i, j], end='\t')
            print()
        print('---------------------------')
    
    def step(self, action):
        node_ = Node(self.puzzle)