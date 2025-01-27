import collections
import numpy as np

Experience = collections.namedtuple('Experience',field_names=['state', 'action', 'reward', 'done', 'next_state'])

class ExperienceReplay:
    def __init__(self, capacity, random_seed):
        self.buffer = collections.deque(maxlen=capacity)
        np.random.seed(random_seed)
        
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
  
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer),batch_size,replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states,dtype=np.float32), np.array(actions,dtype=np.int64), np.array(rewards,dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states,dtype=np.float32)