import numpy as np
from multiprocessing.managers import BaseManager

# This is necessary to share the DeepNet instance among the learners.
class DnnManager(BaseManager): pass

def Manager():
    m = DnnManager()
    m.start()
    return m

class DeepNet:
    
    def __init__(self, size):
        self.size = size
        self.shared = np.zeros((size, size), dtype=float)
    
    def get_mtx(self):
        return self.shared
        
    def dnn_size(self):
        return self.size
    
    def deep_copy(self, target):
        
        if target.shape == self.shared.shape:
            for row in range(0, target.shape[0]):
                for col in range(0, target.shape[1]):
                    target[row, col] = self.shared[row, col]
                    
    def async_update(self, update):
        
        if update.shape == self.shared.shape:
            self.shared += update

            
    def print_mtx(self):
        print (self.shared)
        
        
DnnManager.register('DeepNet', DeepNet)