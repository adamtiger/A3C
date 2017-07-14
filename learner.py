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
    net = dnn.Manager().DeepNet(num_actions)
    temp_env.close()
    
    prms_pi = net.get_parameters_pi()
    prms_v = net.get_parameters_v()
    
    return [prms_pi, prms_v]

def execute_agent(learner_id, atari_env, t_max, T_max, C, eval_num, gamma):
    agent = create_agent(atari_env, t_max, T_max, C, eval_num, gamma)
    agent.run(learner_id)
        
def create_agent(atari_env, t_max, T_max, C, eval_num, gamma):
    return Agent(atari_env, t_max, T_max, C, eval_num, gamma)
    
def create_agent_for_evaluation():
    
    # read the json with data (environemnt name and dnn model)
    
    meta_data = logger.read_metadata()
    atari_name = meta_data[1]
    
    agent = Agent(atari_name, 10000, 0, 0, 0, 0)
    logger.load_model(agent.get_net())
    
    return agent

# During a game attempt, a sequence of observation are generated.
# The last four always forms the state. Rewards and actions also saved.
class Queue:
    
    def __init__(self, max_size):
        self.size = max_size + 1
        
        self.observations = np.ndarray((self.size, 84, 84))
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
        self.rewards[self.last_idx] = reward
        self.actions[self.last_idx] = action
        
        self.is_last_terminal = done
        
    def get_recent_state(self):
        if self.last_idx > 2:
            return self.observations[self.last_idx-3:self.last_idx+1,:,:]
        return None
        
    def get_state_at(self, idx):
        if idx > 2:
            return self.observations[idx-3:idx+1,:,:]
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
    
    # Cropping the playing area. The shape based on empirical decision.
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
    
def env_reset(env, queue):
    queue.queue_reset()
    obs = env.reset()
    queue.add(process_img(obs), 0, 0, False)
    return queue.get_recent_state()
    
def env_step(env, queue, action):
    obs, rw, done, _ = env.step(action)
    queue.add(process_img(obs), rw, action, done)
    return queue.get_recent_state()


class Agent:
    
    def __init__(self, env_name, t_max, T_max, C, eval_num, gamma):
        
        self.t_start = 0
        self.t = 0
        self.t_max = t_max
        
        self.T = 0
        self.T_max = T_max
        
        self.C = C
        self.eval_num = eval_num
        self.gamma = gamma
        
        self.is_terminal = False
        
        self.queue = Queue(t_max)
        self.env = gym.make(env_name)
        self.net = dnn.DeepNet(self.env.action_space.n)
        self.s_t = env_reset(self.env, self.queue)
        
        self.R = 0
        self.sign = False
        
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
            
            self.calculate_gradients()
            
            self.sync_update()
            
            if self.T % self.C == 0:
                self.evaluate_during_training()
    
    # IMPLEMENTATIONS FOR the FUNCTIONS above
        
    def synchronize_dnn(self):
        lock.acquire()
        try:
            self.net.synchronize_net(shared)
        finally:
            lock.release()
        
    def play_game_for_a_while(self):
    
        self.t_start = self.t
        
        self.epsilon = (1.0 - 0.1)*5/self.T_max *self.T
        
        self.s_t = env_reset(self.env, self.queue)
        
        while not (self.is_terminal or self.t - self.t_start == self.t_max):
            self.t += 1
            self.T += 1
            print (self.t)
            action = dnn.action_with_exploration(self.net, self.s_t, self.epsilon)
            self.s_t = env_step(self.env, self.queue, action)
            self.is_terminal = self.queue.get_is_last_terminal()
            if self.T % self.C == 0:
                self.sign = True
        
    def set_R(self):
        if self.is_terminal:
            self.R = 0
        else:
            self.R = self.net.state_value(self.s_t)
        
    def calculate_gradients(self):
        
        idx = self.queue.get_last_idx()
        while idx > 3:
            state = self.queue.get_state_at(idx)
            reward = self.queue.get_reward_at(idx)
            action = self.queue.get_action_at(idx)
            
            self.R = reward + self.gamma * self.R
            self.net.train_net(state, action, self.R)
            
            idx = idx-1
        
        state = self.queue.get_state_at(idx)
        reward = self.queue.get_reward_at(idx)
        action = self.queue.get_action_at(idx)
            
        self.R = reward + self.gamma * R
        self.diff = self.net.train_net(state, action, R)
            
        if self.sign:
            logger.log_losses(0, self.T, self.learner_id) #!
            self.sign = False
        
    def sync_update(self):
        lock.acquire()
        try:
            self.net.sync_update(shared, self.diff)
        finally:
            lock.release()
        
    def evaluate_during_training(self):
        
        print ('Evaluation at: ' + str(self.T))
        
        for rnd in range(self.eval_num):
            state = env_reset(self.env, self.queue)
            finished = False
            cntr = 0
            rewards = []
            while not (finished or cntr == self.t_max):
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
        while not (finished or cntr == self.t_max):
            env.render()
            action = dnn.action(self.net, state)
            state = env_step(env, self.queue, action)
            rewards.append(self.queue.get_recent_reward())
                
            finished = self.queue.get_is_last_terminal()
            cntr += 1  
         
        # Representing the results. 
        print ('The collected rewards over duration:')
        total_rw = 0.0
        for x in range(len(rewards)):
            total_rw += rewards[x]
        print (total_rw)
        