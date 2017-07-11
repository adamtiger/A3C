import os
import shutil
import json
import time

root = 'files'
path_rewards = 'files/rewards/'
path_losses = 'files/losses/'
path_meta = 'files/metadata.json'
path_model = 'files/model.hf5'

def create_folders(atari_name, cores, tmax, Tmax, C):
    if os.path.exists(root):
        # Delete if exists.
        print ('The folder named files is deleted!')
        shutil.rmtree(root)
        
    # Create the new folders.
    os.makedirs(path_rewards)
    os.makedirs(path_losses)
        
    metadata = [time.strftime("%d/%m/%y"), atari_name, str(cores), str(tmax), str(Tmax), str(C)]
    with open(path_meta, "w") as f:
        f.write(json.dumps(metadata))

def log_rewards(rewards, iteration, learner_id, rnd):
    file_name = path_rewards + "rwd_" + str(iteration) + "_" + str(learner_id) + "_" + str(rnd) + ".json"
    with open(file_name, "w") as f:
        f.write(json.dumps(rewards))
    
def log_losses(loss, iteration, learner_id):
    file_name = path_losses + "loss_" + str(iteration) + "_" + str(learner_id) + "_.json"
    with open(file_name, "w") as f:
        f.write(json.dumps(loss))
    
