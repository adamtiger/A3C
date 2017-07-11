import numpy as np
import scipy.misc as sc
import gym
import dnn
import logger

# In case of Pool the Lock object can not be passed in the initialization argument.
# This is the solution
lock = 0
def init_lock(l):
    global lock
    lock = l

# Easier to call these functions from other modules.

def execute_agent(thread_id, atari_env, t_max, T_max, C, shared):
    agent = create_agent(atari_env, t_max, T_max, C, shared)
    agent.run(thread_id)

def create_shared():
    return dnn.Manager().DeepNet(4)
        
def create_agent(atari_env, t_max, T_max, C, shared):
    return Agent(atari_env, t_max, T_max, C, shared)
    
def create_agent_for_evaluation(file_name):
    
    # read the json with data (environemnt name and dnn model)
    
    agent = Agent('Breakout-v0', 0, 0, 0, None)
    agent.read_model(file_name)
    
    return agent

# During a game attempt, a sequence of observation are generated.
# The last four always forms the state. Rewards and actions also saved.
class Queue:
    
    def __init__(self, max_size):
        self.size = max_size
        
        self.observations = np.ndarray((self.size, 84, 84, 1))
        self.rewards = np.ndarray((self.size))
        self.actions = np.ndarray((self.size))
        
        self.stacked = np.ndarray((84, 84, 4))
        
        self.last_idx = 0
        self.is_last_terminal = False
    
    def get_last_idx(self):
        return self.last_idx
        
    def get_is_last_terminal(self):
        return self.is_last_terminal
    
    def queue_reset(self):
        self.observations.fill(0)
        self.rewards.fill(0)
        self.actions.fill(0)
        self.last_idx = 0
        self.is_last_terminal = False
        
    def add(self, observation, reward, action, done):
        self.observations[self.last_idx, :, :, :] = observation[:,:,:]
        self.rewards[self.last_idx] = reward
        self.actions[self.last_idx] = action
        
        self.last_idx += 1
        self.is_last_terminal = done
        
    def get_recent_state(self):
        if self.last_idx > 2:
            return self.observations[self.last_idx-3:self.last_idx+1,:,:,:]
        return None
        
    def get_state_at(self, idx):
        if idx > 2:
            return self.observations[idx-3:idx+1,:,:,:]
        return None
    
    def get_reward_at(self, idx):
        return self.rewards[idx]
    
    def get_action_at(self, idx):
        return self.actions[idx]


# Preprocessing of the raw frames from the game.

def process_img(observation):

    # Input: the raw frame as a list with shape (210, 160, 3)
    # https://gym.openai.com/envs/Breakout-v0 
    img = np.array(observation)
    
    # Cropping the playing area. The shape based on empirical decision.
    img_cropped = np.zeros((185, 160, 3))
    img_cropped[:,:,:] = img[16:201,:,:]
    
    # Mapping from RGB to gray scale. (Shape remains unchanged.)
    # Y = (2*R + 5*G + 1*B)/8
    img_cropped_gray = np.zeros((img_cropped.shape[0], img_cropped.shape[1], 1))
    img_cropped_gray = (2*img_cropped[:,:,0] + 5*img_cropped[:,:,1] + img_cropped[:,:,2])/8.0
    
    # Rescaling image to 84x84x1.
    img_cropped_gray_resized = np.zeros((84,84,1))
    img_cropped_gray_resized[:,:,0] = sc.imresize(img_cropped_gray, (84,84,1), interp='bilinear', mode=None)
    
    # Saving memory. Colors goes from 0 to 255.
    img_final = np.uint8(img_cropped_gray_resized)
    
    return img_final
    
def env_reset(env, queue):
    queue.queue_reset()
    obs = env.reset()
    return process_img(obs)
    
def env_step(env, queue, action):
    obs, rw, done, _ = env.step(action)
    queue.add(process_img(obs), rw, action, done)
    return queue.get_recent_state()


class Agent:
    
    def __init__(self, env_name, t_max, T_max, C, shared):
        
        self.t_start = 0
        self.t = 0
        self.t_max = t_max
        
        self.T = 0
        self.T_max = T_max
        
        self.C = C
        
        self.shared = shared
        
        self.is_terminal = False
        
        self.queue = Queue(t_max)
        self.env = gym.make(env_name)
        self.s_t = env_reset(self.env, self.queue)
        
        self.gradients = np.zeros((shared.dnn_size(), shared.dnn_size())) #!
        self.own = dnn.DeepNet(shared.dnn_size()) #!
        
        self.R = 0
    
    # For details: https://arxiv.org/abs/1602.01783
    def run(self, thread_id):
        
        self.thread_id = thread_id
        
        while self.T < self.T_max:

            self.reset_gradients()
            self.synchronize_dnn()
            self.t_start = self.t
            
            self.play_game_for_a_while()
            
            self.set_R()
            
            self.calculate_gradients()
            
            self.async_update()
            
            if self.T % self.C == 0:
                self.evaluate_during_training()
    
    # IMPLEMENTATIONS FOR the FUNCTIONS above
                
    def reset_gradients(self):
        pass
        
    def synchronize_dnn(self):
        lock.acquire()
        try:
            self.shared.deep_copy(self.own.get_mtx())
        finally:
            lock.release()
        
    def play_game_for_a_while(self):
    
        self.t_start = self.t
        
        self.s_t = env_reset(self.env, self.queue)
        
        while not (self.is_terminal or self.t - self.t_start == self.t_max):
            self.t += 1
            self.T += 1
            action = dnn.action_with_exploration(self.own, self.s_t)
            self.s_t = env_step(self.env, self.queue, action)
            self.is_terminal = self.queue.get_is_last_terminal()
            if self.T % 100 == 0:
                print (self.T/100.0)
        
    def set_R(self):
        if self.is_terminal:
            self.R = 0
        else:
            self.R = dnn.value(self.own, self.s_t)
        
    def calculate_gradients(self):
        self.gradients.fill(1)
    
    def async_update(self):
        lock.acquire()
        try:
            self.shared.async_update(self.gradients)
        finally:
            lock.release()
        
    def evaluate_during_training(self):
        pass
    
    def evaluate(self):
        pass   
        
    # Read and write the model.
    
    def save_model(self, file_name):
        pass
        
    def read_model(self, file_name):
        pass