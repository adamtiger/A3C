import gym
import dnn
import logger

def create_shared():
    pass
        
def create_agent(atari_env, t_max, T_max, C, shared):
    return Agent(atari_env, t_max, T_max, C, shared)
    
def create_agent_for_evaluation(file_name):
    
    # read the json with data (environemnt name and dnn model)
    
    agent = Agent('Breakout-v0', 0, 0, 0, None)
    agent.read_model(file_name)
    
    return agent

class Agent:
    
    def __init__(self, env_name, t_max, T_max, C, shared):
        
        self.env = gym.make(env_name)
        self.s_t = self.env.reset()
        
        self.t_start = 0
        self.t = 0
        self.t_max = t_max
        
        self.T = 0
        self.T_max = T_max
        
        self.C = C
        
        self.shared = shared
        
    def run(self, lock, thread_id):
        
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
                
    def reset_gradients(self):
        pass
        
    def synchronize_dnn(self):
        pass
        
    def play_game_for_a_while(self):
        self.t += 1
        self.T += 1
        pass
    
    def set_R(self):
        pass
        
    def calculate_gradients(self):
        pass
    
    def async_update(self):
        pass
        
    def evaluate_during_training(self):
        pass
    
    def evaluate(self):
        pass   
        
    # Read and write the model.
    
    def save_model(self, file_name):
        pass
        
    def read_model(self, file_name):
        pass