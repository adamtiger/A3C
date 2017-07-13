import numpy as np
from multiprocessing.managers import BaseManager

import cntk
from cntk.device import try_set_default_device, cpu
from cntk.layers import Convolution2D, Dense, Sequential
from cntk.learners import adam, learning_rate_schedule, momentum_schedule

# This is necessary to share the DeepNet instance among the learners.
class DnnManager(BaseManager): pass

def Manager():
    m = DnnManager()
    m.start()
    return m

# Set CPU as device for the neural network.
try_set_default_device(cpu())

class DeepNet:
    
    def __init__(self, num_actions): # (*)
        self.num_actions = num_actions
        self.shared = np.zeros((num_actions, num_actions), dtype=float)
        
    def build_model(self):
        
        # Defining the input and output variables.
        stacked_frames = cntk.input_variable((84, 84, 4))
        
        # Creating the shared functions. (The common part of the two NNs.)
        conv1 = Convolution2D((8, 8), num_filters = 16, pad = False, strides=4, activation=cntk.relu)
        conv2 = Convolution2D((4, 4), num_filters = 32, pad = False, strides=2, activation=cntk.relu)
        dense = Dense(256, activation=cntk.relu)
        shared_funcs = Sequential([conv1, conv2, dense])
        
        # Creating the value approximator extension.
        v = Sequential([shared_funcs, Dense(1)])
        parameters_v = v(stacked_frames)
        
        # Creating the policy approximator extension.
        pi = Sequential([shared_funcs, Dense(self.num_actions, activation=cntk.softmax)])
        parameters_pi = pi(stacked_frames)
        
        self.model = [pi, parameters_pi, v, parameters_v] 
        
    def build_trainer(self):
        
        # Set the learning rate, and the momentum parameters for the Adam optimizer.
        
        lr = learning_rate_schedule(0.00025, UnitType.minibatch)
        beta1 = momentum_schedule(0.9)
        beta2 = momentum_schedule(0.99)
        
        # Calculate the losses.
        
        loss_on_v
        loss_on_pi
        
        # Create the trainiers.
        
        trainer_v = cntk.Trainer(self.model[3], (loss_on_v), [adam(lr, beta1, beta2)])
        trainer_pi = cntk.Trainer(self.model[1], (loss_on_pi), [adam(lr, beta1, beta2)])
        
        self.trainer = [trainer_pi, trainer_v]
        
        
    def dnn_actions(self):
        return self.num_actions
    
    def synchronize_net(self, net): 
        
                    
    def sync_update(self, update):


        
DnnManager.register('DeepNet', DeepNet)


# Functions to generate the next actions
import random as r
def action(net, state): # Take into account None as input -> generate random actions
    n = net.dnn_actions()
    return r.randint(0, n-1)
    
def action_with_exploration(net, state): # Take into account None as input -> generate random actions
    return 0                             # Epsilon-greedy is a right approach.
    
def value(net, state):
    return 0