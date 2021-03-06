import numpy as np
import scipy.misc as sc
import gym
from gym import wrappers
import dnn
import logger

# In case of Pool the Lock object can not be passed in the initialization argument.
# This is the solution
lock = 0
shared = 0
def init_lock_shared(l, sh):
    global lock
    global shared
    lock = l
    shared = sh

# Easier to call these functions from other modules.
def create_shared(env_name):
    temp_env = gym.make(env_name)
    num_actions = temp_env.action_space.n
    net = dnn.DeepNet(num_actions, 0)
    temp_env.close()
    
    prms_pi = net.get_parameters_pi()
    prms_v = net.get_parameters_v()
    
    return [prms_pi, prms_v]

def execute_agent(learner_id, atari_env, t_max, game_length, T_max, C, eval_num, gamma, lr):
    agent = create_agent(atari_env, t_max, game_length, T_max, C, eval_num, gamma, lr)
    agent.run(learner_id)
        
def create_agent(atari_env, t_max, game_length, T_max, C, eval_num, gamma, lr):
    return Agent(atari_env, t_max, game_length, T_max, C, eval_num, gamma, lr)
    
def create_agent_for_evaluation():
    
    # read the json with data (environemnt name and dnn model)
    
    meta_data = logger.read_metadata()
    atari_name = meta_data[1]
    
    agent = Agent(atari_name, 50000, 50000, 0, 0, 0, 0, 0) 
    logger.load_model(agent.get_net())
    
    return agent

# During a game attempt, a sequence of observation are generated.
# The last four always forms the state. Rewards and actions also saved.
class Queue:
    
    def __init__(self, max_size):
        self.size = max_size
        
        self.observations = np.ndarray((self.size, 84, 84), dtype=np.uint8) # save memory with uint8
        self.rewards = np.ndarray((self.size))
        self.actions = np.ndarray((self.size))
        
        self.last_idx = -1
        self.is_last_terminal = False
    
    def get_last_idx(self):
        return self.last_idx
        
    def get_is_last_terminal(self):
        return self.is_last_terminal
    
    def queue_reset(self):
        self.observations.fill(0)
        self.rewards.fill(0)
        self.actions.fill(0)
        self.last_idx = -1
        self.is_last_terminal = False
        
    def add(self, observation, reward, action, done):
        self.last_idx += 1
        self.observations[self.last_idx, :, :] = observation[0,:,:]
        if reward > 1.0:
            reward = 1.0 # reward clipping
        self.rewards[self.last_idx] = reward
        self.actions[self.last_idx] = action
        
        self.is_last_terminal = done
        
    def get_recent_state(self):
        if self.last_idx > 2:
            return np.float32(self.observations[self.last_idx-3:self.last_idx+1,:,:])
        return None
        
    def get_state_at(self, idx):
        if idx > 2:
            return np.float32(self.observations[idx-3:idx+1,:,:])
        return None
    
    def get_reward_at(self, idx):
        return self.rewards[idx]
        
    def get_recent_reward(self):
        return self.rewards[self.last_idx]
    
    def get_action_at(self, idx):
        return self.actions[idx]


# Preprocessing of the raw frames from the game.

def process_img(observation):

    # Input: the raw frame as a list with shape (210, 160, 3)
    # https://gym.openai.com/envs/Breakout-v0 
    input_img = np.array(observation)
    
    # Reshape input to meet with CNTK expectations.
    img = np.reshape(input_img, (3, 210, 160))
    
    # Cropping the playing area. The shape is based on empirical decision.
    img_cropped = np.zeros((3, 185, 160))
    img_cropped[:,:,:] = img[:, 16:201,:]
    
    # Mapping from RGB to gray scale. (Shape remains unchanged.)
    # Y = (2*R + 5*G + 1*B)/8
    img_cropped_gray = np.zeros((1, img_cropped.shape[0], img_cropped.shape[1]))
    img_cropped_gray = (2*img_cropped[0,:,:] + 5*img_cropped[1,:,:] + img_cropped[2,:,:])/8.0
    
    # Rescaling image to 1x84x84.
    img_cropped_gray_resized = np.zeros((1,84,84))
    img_cropped_gray_resized[0,:,:] = sc.imresize(img_cropped_gray, (1,84,84), interp='bilinear', mode=None)
    
    # Saving memory. Colors goes from 0 to 255.
    img_final = np.uint8(img_cropped_gray_resized)
    
    return img_final

# Functions to avoid temporary coupling.
def env_reset(env, queue):
    queue.queue_reset()
    obs = env.reset()
    queue.add(process_img(obs), 0, 0, False)
    return queue.get_recent_state() # should return None
    
def env_step(env, queue, action):
    obs, rw, done, _ = env.step(action)
    queue.add(process_img(obs), rw, action, done)
    return queue.get_recent_state()


class Agent:
    
    def __init__(self, env_name, t_max, game_length, T_max, C, eval_num, gamma, lr):
        
        self.t_start = 0
        self.t = 0
        self.t_max = t_max
        
        self.game_length = game_length
        
        self.T = 0
        self.T_max = T_max
        
        self.C = C
        self.eval_num = eval_num
        self.gamma = gamma
        
        self.is_terminal = False
        
        self.queue = Queue(game_length) 
        self.env = gym.make(env_name)
        self.net = dnn.DeepNet(self.env.action_space.n, lr)
        self.s_t = env_reset(self.env, self.queue)
        
        self.R = 0
        self.signal = False
        
        self.diff = []
        self.epsilon = 1.0
    
    def get_net(self):
        return self.net
    
    # For details: https://arxiv.org/abs/1602.01783
    def run(self, learner_id):
        
        self.learner_id = learner_id
        
        while self.T < self.T_max:

            self.synchronize_dnn()
            
            self.play_game_for_a_while()
            
            self.set_R()
            
            # According to the article the gradients should be calculated.
            # Here: The parameters are updated and the differences are added to the shared NN's.
            self.calculate_gradients()
            
            self.sync_update() # Syncron update instead of asyncron!
            
            if self.signal:
                self.evaluate_during_training()
                self.signal = False
    
    # IMPLEMENTATIONS FOR the FUNCTIONS above
        
    def synchronize_dnn(self):
        lock.acquire()
        try:
            self.net.synchronize_net(shared) # the shared parameters are copied into 'net'
        finally:
            lock.release()
        
    def play_game_for_a_while(self):
    
        if self.is_terminal:
            self.s_t = env_reset(self.env, self.queue)
            self.t = 0
            self.is_terminal = False
            
        self.t_start = self.t
        
        self.epsilon = max(0.1, 1.0 - (1.0 - 0.1)*5/self.T_max*self.T) # first decreasing, then it is constant
        
        while not (self.is_terminal or self.t - self.t_start == self.t_max):
            self.t += 1
            self.T += 1
            action = dnn.action_with_exploration(self.net, self.s_t, self.epsilon)
            self.s_t = env_step(self.env, self.queue, action)
            self.is_terminal = self.queue.get_is_last_terminal()
            if self.T % self.C == 0: # log loss when evaluation happens
                self.signal = True
            if self.T % 5000 == 0:
                print('Actual iter. num.: ' + str(self.T))
        
    def set_R(self):
        if self.is_terminal:
            self.R = self.net.state_value(self.s_t)
            self.R[0][0] = 0.0 # Without this, error dropped. special format is given back.
        else:
            self.R = self.net.state_value(self.s_t)
        
    def calculate_gradients(self):
        
        idx = self.queue.get_last_idx()
        while idx > 3: # the state is 4 pieces of frames stacked together -> at least 4 frames are necessary
            state = self.queue.get_state_at(idx)
            reward = self.queue.get_reward_at(idx)
            action = self.queue.get_action_at(idx)
            self.R = reward + self.gamma * self.R
            self.net.train_net(state, action, self.R, False)
            
            idx = idx-1
        
        # At the last training step the differences should be saved
        state = self.queue.get_state_at(idx)
        reward = self.queue.get_reward_at(idx)
        action = self.queue.get_action_at(idx)
            
        self.R = reward + self.gamma * self.R
        self.diff = self.net.train_net(state, action, np.float32(self.R), True)
            
        if self.signal:
            logger.log_losses(self.net.get_last_avg_loss(), self.T, self.learner_id)
        
    def sync_update(self):
        lock.acquire()
        try:
            self.net.sync_update(shared, self.diff)
        finally:
            lock.release()
        
    def evaluate_during_training(self):
        
        print ('Evaluation at: ' + str(self.T))
        
        for rnd in range(self.eval_num): # Run more game epsiode to get more robust result for performance
            state = env_reset(self.env, self.queue)
            finished = False
            cntr = 0
            rewards = []
            while not (finished or cntr == self.game_length):
                action = dnn.action(self.net, state)
                state = env_step(self.env, self.queue, action)
                rewards.append(self.queue.get_recent_reward())
                
                finished = self.queue.get_is_last_terminal()
                cntr += 1
            
            logger.log_rewards(rewards, self.T, self.learner_id, rnd)
            
    def evaluate(self):
        
        print ('Start evaluating.')
        env = wrappers.Monitor(self.env, 'videos', force=True)
        state = env_reset(env, self.queue)
        finished = False
        cntr = 0
        rewards = []
        while not (finished or cntr == self.game_length):
            env.render()
            action = dnn.action(self.net, state)
            state = env_step(env, self.queue, action)
            rewards.append(self.queue.get_recent_reward())
                
            finished = self.queue.get_is_last_terminal()
            cntr += 1  
         
        # Representing the results. 
        print ('The collected rewards over duration:')
        total_rw = 0.0
        for x in rewards:
            total_rw += x
        print (total_rw)

    def save_model(self, shared_params, path_model_pi, path_model_v):
        self.net.synchronize_net(shared_params) # copy the parameters into the recently created agent's netork
        self.net.save_model(path_model_pi, path_model_v)
        
