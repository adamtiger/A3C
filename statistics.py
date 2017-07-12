import json
import os
import matplotlib.pyplot as plt


path_rewards = 'files/rewards/'
path_losses = 'files/losses/'

# CREATE PLOTS and draw some STATISTICAL DATA

def collect_reward_file_names():
    total_rewards = {} 
    list_of_names = os.listdir(path_rewards)
    
    for idx in range(0, len(list_of_names)):
        attributes = list_of_names[idx].split(".")[0] # Cut the json part from the string.
        attributes = attributes.split("/") # Cut the string part with the numbers.
        attributes = attributes[0].split("_")[1:4] # Contains the iteration, learner_id and rnd values.
        
        iteration = int(attributes[0])
        learner_id = int(attributes[1])
        
        file_name = path_rewards + list_of_names[idx]
        with open(file_name, "r") as f:
            rewards = json.load(f)
        
        summa = 0.0
        for x in rewards:
            summa += x
            
        if learner_id not in list(total_rewards):
            total_rewards[learner_id] = {}
        if iteration in list(total_rewards[learner_id]):
             total_rewards[learner_id][iteration].append(summa)
        else:
            total_rewards[learner_id][iteration] = [summa]
    
    return total_rewards
    
def calculate_avg_rewards(total_rewards, learner_id):
    
    iterations = list(total_rewards[learner_id])
    
    # Calculating the averages.
    avg_rewards = [0] * len(iterations)
    
    for it in range(len(iterations)):
        for rw in total_rewards[learner_id][iterations[it]]:
            avg_rewards[it] += rw
        avg_rewards[it] = avg_rewards[it]/len(total_rewards[learner_id][it])
        
    return [avg_rewards, iterations]
    
def calculate_max_rewards(total_rewards, learner_id):
    
    iterations = list(total_rewards[learner_id])
    
    # Calculating the averages.
    max_rewards = [0] * len(iterations)
    
    for it in range(len(iterations)):
        max_rewards[it] = total_rewards[learner_id][iterations[it]][0]
        for rw in total_rewards[learner_id][iterations[it]]:
            if max_rewards[it] < rw:
                max_rewards[it] = rw
        
    return [max_rewards, iterations]
    
def calculate_min_rewards(total_rewards, learner_id):
    
    iterations = list(total_rewards[learner_id])
    
    # Calculating the averages.
    min_rewards = [0] * len(iterations)
    
    for it in range(len(iterations)):
        min_rewards[it] = total_rewards[learner_id][iterations[it]][0]
        for rw in total_rewards[learner_id][iterations[it]]:
            if min_rewards[it] > rw:
                min_rewards[it] = rw
        
    return [min_rewards, iterations]

def collect_loss_file_names():
    total_losses = {} 
    list_of_names = os.listdir(path_losses)
    
    for idx in range(0, len(list_of_names)):
        attributes = list_of_names[idx].split(".")[0] # Cut the json part from the string.
        attributes = attributes.split("/") # Cut the string part with the numbers.
        attributes = attributes[0].split("_")[1:3] # Contains the iteration, learner_id.
        
        iteration = int(attributes[0])
        learner_id = int(attributes[1])
        
        file_name = path_losses + list_of_names[idx]
        with open(file_name, "r") as f:
            loss = json.load(f)
            
        if learner_id not in list(total_losses):
            total_losses[learner_id] = {}
        if iteration not in list(total_losses[learner_id]):
             total_losses[learner_id][iteration] = loss
    
    return total_losses

def loss_of_learner(total_losses, learner_id):
    
    iterations = list(total_losses[learner_id])
    losses = list(total_losses[learner_id].values())
    
    return [losses, iterations]
    
def draw_simple_graphs(num, x, y, y_label):
    
    plt.figure(num)
    plt.plot(x, y, 'ro')
    plt.xlabel('Number of actions')
    plt.ylabel(y_label)
    plt.show()