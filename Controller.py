import os

import numpy as np


class Controller:
    def __init__(self, weights = None, num_sensors = 12, extra_inputs = 5, num_outputs = 2):
        self.weights = weights
        self.num_sensors = num_sensors
        self.num_inputs = self.num_sensors + extra_inputs
        self.num_outputs = num_outputs
        self.max_speed = 30

        if self.weights is None:
            self.generate_weights()


    def generate_weights(self):
        self.weights = np.random.uniform(-1, 1, (self.num_outputs, self.num_inputs))

    def out(self, input, v_left, v_right, pos_x, pos_y, angle):
        full_in = np.array(input + [pos_x, pos_y, angle, v_left, v_right])
        outputs = np.tanh(np.dot(self.weights, full_in))*self.max_speed
        return outputs[0], outputs[1]

    def get_genome(self):
        return self.weights.flatten()

    def set_genome(self, genome):
        self.weights = np.array(genome).reshape(self.num_outputs, self.num_inputs)

    def save(self, dirpath, filename):
        os.makedirs(dirpath, exist_ok=True)
        filepath = os.path.join(dirpath, filename)
        np.save(str(filepath), self.get_genome())

    def load(self, filepath):
        self.set_genome(np.load(filepath))