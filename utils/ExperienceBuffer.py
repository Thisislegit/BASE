
import numpy as np
import queue
import random
from collections import deque



class ExperienceBuffer:
    def __init__(self, capacity):
        self.memory = [[] for i in range(capacity)]
        self.capacity = capacity
        self.memory_counter = 0

    def __len__(self):
        return len(self.buffer)

    def check_state(self):
        if self.memory_counter < self.capacity:
            return False
        else:
            return True

    def append(self, experience):
        index = self.memory_counter % self.capacity
        self.memory[index] = experience
        self.memory_counter += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size if batch_size <= self.memory_counter else self.memory_counter)


class PrioritizedReplayBuffer():
    def __init__(self, maxlen):
        self.capacity = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.memory_counter = 0

    def check_state(self):
        if self.memory_counter < self.capacity:
            return False
        else:
            return True

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))
        self.memory_counter += 1

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1 / len(self.buffer) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        # samples = np.array(self.buffer)[sample_indices]
        samples = [self.buffer[i] for i in sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        # return map(list, zip(*samples)), importance, sample_indices
        return samples, importance, sample_indices

    def set_priorities(self, indices, errors, offset=1e-5):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

if __name__ == '__main__':
    p = PrioritizedReplayBuffer(3)
    p.add([1])
    p.add([2])
    p.add([3])
    p.set_priorities([0,1], [100, 10])
    print(p.buffer)
    print(p.priorities)
    print(p.sample(3, 0))