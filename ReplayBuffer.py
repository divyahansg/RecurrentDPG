# Inspired by: https://github.com/yanpanlau/DDPG-Keras-Torcs/blob/master/ReplayBuffer.py

from collections import deque
import random

class ReplayBuffer:
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.count = 0
        self.buffer = deque()

    def getCount(self):
        return self.count

    def size(self):
        return self.bufferSize

    def add(self, sequence):
        if self.count < self.bufferSize:
            self.buffer.append(sequence)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(sequence)

    def getBatch(self, batchSize):
        # Randomly sample batchSize examples
        if self.count < batchSize:
            return random.sample(self.buffer, self.count)
        else:
            return random.sample(self.buffer, batchSize)

    def erase(self):
        self.buffer = deque()
        self.count = 0
