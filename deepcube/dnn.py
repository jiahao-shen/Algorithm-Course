import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten


class DNN(object):

    def __init__(self, n):
        self.n = n
        self.learning_rate = 1e-3
        self.estimate_model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Dense(units=512, input_dim=self.n * self.n))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=1, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def train(self, x, y):
        callback = self.target_model.fit(x, y, batch_size=32, epochs=10, verbose=0)
        return np.average(callback.history['loss'])

    def estimate(self, state):
        return self.estimate_model.predict(state)[0]

    def predict(self, state):
        return self.target_model.predict(state)[0]

    def update(self):
        self.estimate_model.set_weights(self.target_model.get_weights())

    def save(self, file):
        self.target_model.save_weights(file)

    def load(self, file):
        self.target_model.load_weights(file)
