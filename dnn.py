import numpy as np
from multiprocessing.managers import BaseManager

# This is necessary to share the DeepNet instance among the learners.
class DnnManager(BaseManager): pass

def Manager():
    m = DnnManager()
    m.start()
    return m

class DeepNet:
    
    def __init__(self, num_actions): # (*)
        self.num_actions = num_actions
        self.shared = np.zeros((num_actions, num_actions), dtype=float)
    
    def get_mtx(self):
        return self.shared
        
    def dnn_size(self):
        return self.num_actions
    
    def synchronize_net(self, net): # (*)
        
        if net.get_mtx().shape == self.shared.shape:
            for row in range(0, 5):
                for col in range(0, 5):
                    net.get_mtx()[row, col] = self.shared[row, col]
                    
    def sync_update(self, update): # (*)
        if update.shape == self.shared.shape:
            self.shared += update

    def print_mtx(self):
        print (self.shared)
        
        
DnnManager.register('DeepNet', DeepNet)


# Functions to generate the next actions
import random as r
def action(net, state):
    n = net.dnn_size()
    return r.randint(0, n-1)
    
def action_with_exploration(net, state):
    return 0
    
def value(net, state):
    return 0