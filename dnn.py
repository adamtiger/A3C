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
        
        self.build_model()
        self.build_trainer()
        
    def build_model(self):
        
        # Defining the input variables for training and evaluation.
        self.stacked_frames = cntk.input_variable((84, 84, 4))
        self.action = cntk.input_variable(self.num_actions)
        self.R = cntk.input_variable(1)
        
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
        
        self.pi = pi
        self.pms_pi = parameters_pi
        self.v = v
        self.pms_v = parameters_v 
        
    def build_trainer(self):
        
        # Set the learning rate, and the momentum parameters for the Adam optimizer.
        
        lr = learning_rate_schedule(0.00025, UnitType.minibatch)
        beta1 = momentum_schedule(0.9)
        beta2 = momentum_schedule(0.99)
        
        # Calculate the losses.
        
        loss_on_v = cntk.squared_error(self.R, self.v)
        
        pi_a_s = cntk.times_transpose(cntk.log(self.pi), self.action)
        loss_on_pi = cntk.times(pi_a_s, cntk.minus(self.R, self.v))
        
        # Create the trainiers.
        
        trainer_v = cntk.Trainer(self.pms_v, (loss_on_v), [adam(lr, beta1, beta2)])
        trainer_pi = cntk.Trainer(self.pms_pi, (loss_on_pi), [adam(lr, beta1, beta2)])
        
        self.trainer_pi = trainer_pi
        self.trainer_v = trainer_v
    
    def train_net(self, state, action, R):
        
        action_as_array = np.zeros(self.num_actions)
        action_as_array[action] = 1
        
        self.trainer_pi.train_minibatch({self.stacked_frames: state, self.action: action_as_array, self.R = R})
        self.trainer_v.train_minibatch({self.stacked_frames: state, self.R = R})
    
    def dnn_actions(self):
        return self.num_actions
    
    def synchronize_net(self, net): 
        cntk.assign(net.get_parameters_pi(), self.pms_pi)
        cntk.assign(net.get_parameters_v(), self.pms_v)
                    
    def sync_update(self, update_pi, update_v):
        cntk.assign(self.pms_pi, cntk.plus(self.pms_pi, update_pi)).eval()
        cntk.assign(self.pms_v, cntk.plus(self.pms_v, update_v)).eval()
    
    def get_parameters_pi(self):
        return self.pms_pi
        
    def get_parameters_v(self):
        return self.pms_v
        
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