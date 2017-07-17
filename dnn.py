import numpy as np
import cntk
from cntk.device import try_set_default_device, cpu
from cntk.layers import Convolution2D, Dense, Sequential
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType

# Set CPU as device for the neural network.
try_set_default_device(cpu())

class DeepNet:
    
    def __init__(self, num_actions, lr):
        self.num_actions = num_actions
        self.lr = lr
        
        self.build_model()
        self.build_trainer()
        
    def build_model(self):
        
        # Defining the input variables for training and evaluation.
        self.stacked_frames = cntk.input_variable((4, 84, 84), dtype=np.float32)
        self.action = cntk.input_variable(self.num_actions)
        self.R = cntk.input_variable(1, dtype=np.float32)
        
        # Creating the shared functions. (The common part of the two NNs.)
        conv1 = Convolution2D((8, 8), num_filters = 16, pad = False, strides=4, activation=cntk.relu)
        conv2 = Convolution2D((4, 4), num_filters = 32, pad = False, strides=2, activation=cntk.relu)
        dense = Dense(256, activation=cntk.relu)
        shared_funcs = Sequential([conv1, conv2, dense])
        
        # Creating the value approximator extension.
        v = Sequential([shared_funcs, Dense(1)])
        
        # Creating the policy approximator extension.
        pi = Sequential([shared_funcs, Dense(self.num_actions, activation=cntk.softmax)])
        
        self.pi = pi(self.stacked_frames)
        self.pms_pi = self.pi.parameters # List of cntk Parameter types (containes the function's parameters)
        self.v = v(self.stacked_frames)
        self.pms_v = self.v.parameters
        

    def build_trainer(self):
        
        # Set the learning rate, and the momentum parameters for the Adam optimizer.
        lr = learning_rate_schedule(self.lr, UnitType.minibatch)
        beta1 = momentum_schedule(0.9)
        #beta2 = momentum_schedule(0.99)
        
        # Calculate the losses.
        loss_on_v = cntk.squared_error(self.R, self.v)
        
        pi_a_s = cntk.times_transpose(cntk.log(self.pi), self.action)
        loss_on_pi = cntk.times(pi_a_s, cntk.minus(self.R, self.v))
        
        # Create the trainiers.
        trainer_v = cntk.Trainer(self.v, (loss_on_v), [adam(self.pms_v, lr, beta1, gradient_clipping_threshold_per_sample=1.0)])
        trainer_pi = cntk.Trainer(self.pi, (loss_on_pi), [adam(self.pms_pi, lr, beta1, gradient_clipping_threshold_per_sample=1.0)])
        
        self.trainer_pi = trainer_pi
        self.trainer_v = trainer_v
    
    def train_net(self, state, action, R, calc_diff):
        
        diff = None
        
        if calc_diff:
            # Save the parameters before a training step.
            self.update_pi = []
            for x in self.pms_pi:
                self.update_pi.append(x.value)
            self.update_v = []
            for x in self.pms_v:
                self.update_v.append(x.value)
        
        # Training part
        action_as_array = np.zeros(self.num_actions)
        action_as_array[int(action)] = 1
        
        self.trainer_pi.train_minibatch({self.stacked_frames: [state], self.action: [action_as_array], self.R: [R]})
        self.trainer_v.train_minibatch({self.stacked_frames: [state], self.R: [R]})
        
        if calc_diff:
            # Calculate the differences between the updated and the original params.
            for idx in range(len(self.pms_pi)):
                self.update_pi[idx] = self.pms_pi[idx].value - self.update_pi[idx]
            for idx in range(len(self.pms_v)):
                self.update_v[idx] = self.pms_v[idx].value - self.update_v[idx]
            
            diff = [self.update_pi, self.update_v]
        
        return diff
        
    def state_value(self, state):
        return self.v.eval({self.stacked_frames: [state]})
    
    def pi_probabilities(self, state):
        return self.pi.eval({self.stacked_frames: [state]})
    
    def get_num_actions(self):
        return self.num_actions
        
    def get_last_avg_loss(self):
        return self.trainer_pi.previous_minibatch_loss_average + self.trainer_v.previous_minibatch_loss_average
    
    def synchronize_net(self, shared): 
        for idx in range(0, len(self.pms_pi)):
            self.pms_pi[idx].value = shared[0][idx]
        for idx in range(0, len(self.pms_v)):
            self.pms_v[idx].value = np.asarray(shared[1][idx])
                    
    def sync_update(self, shared, diff):
        for idx in range(0, len(self.pms_pi)):
            shared[0][idx] += diff[0][idx]
        for idx in range(0, len(self.pms_v)):
            shared[1][idx] += diff[1][idx]
    
    def get_parameters_pi(self):
        pickle_prms_pi = []
        for x in self.pms_pi:
            pickle_prms_pi.append(x.value) # .value gives numpy arrays
        return pickle_prms_pi
        
    def get_parameters_v(self):
        pickle_prms_v = []
        for x in self.pms_v:
            pickle_prms_v.append(x.value) # .value gives numpy arrays
        return pickle_prms_v              # to avoid: can't pickle SwigPyObject
        
    def load_model(self, file_name_pi, file_name_v):
        self.pi.restore(file_name_pi) # load(fn) does different things, it would
        self.v.restore(file_name_v)   # create a new function. It did not work.
        
    def save_model(self, file_name_pi, file_name_v):
        self.pi.save(file_name_pi)
        self.v.save(file_name_v)
        
# Functions to generate the next actions
import random as r
def action(net, state): # Take into account None as input -> generate random actions
    
    act = 0
    n = net.get_num_actions()
    if state is None:
        act = r.randint(0, n-1) 
    else:
        # Decide to explore or not. (In order to avoid the moveless situations.)
        explore = r.randint(0, 1000)
        if explore < 0.05 * 1000:
            act = r.randint(0, n-1)
        else:
            prob_vec = net.pi_probabilities(state)[0] * 1000
            candidate = r.randint(0, 1000)
        
            for i in range(0, n):
                if prob_vec[i] >= candidate:
                    act = i
    
    return act
    
def action_with_exploration(net, state, epsilon): # Take into account None as input -> generate random actions
                                                  # Epsilon-greedy is a right approach.
    act = 0
    n = net.get_num_actions()
    if state is None:
        act = r.randint(0, n-1) 
    else:
        # Decide to explore or not.
        explore = r.randint(0, 1000)
        if explore < epsilon * 1000:
            act = r.randint(0, n-1)
        else:
            prob_vec = net.pi_probabilities(state)[0] * 1000
            candidate = r.randint(0, 1000)
        
            for i in range(0, n):
                if prob_vec[i] >= candidate:
                    act = i
    
    return act
