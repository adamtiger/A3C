from multiprocessing import Process, Lock
import argparse
import learner as lrn

# Read the parameters to initialize the algorithm.
parser = argparse.ArgumentParser(description='A3C algorithm')

parser.add_argument('--atari-env', default='Breakout-v0', metavar='S',
        help='the name of the Atari environment (default:Breakout-v0)')
parser.add_argument('--num-cores', type=int, default=2, metavar='N',
        help='the number of cores should be exploited (default:2)')
parser.add_argument('--t-max', type=int, default='50', metavar='N',
        help='maximal length of a training game (default:50)')
parser.add_argument('--T-max', type=int, default=1000, metavar='N',
        help='the length of the training (default:1000)')
parser.add_argument('--C', type=int, default=25, metavar='N',
        help='the frequency of evaluation during training (default:25)')
parser.add_argument('--train-mode', action='store_true',
        help='training or evaluation')
parser.add_argument('--file-name', default='model.json', metavar='S',
        help='the name of the file where the model should be saved (default:model.json)')

args = parser.parse_args()

# IF TRAIN mode -> train the learners

if (args.train_mode):
    
    # Initialize the whole program

    shared = lrn.create_shared()
    
    # start the processes
    if __name__ == '__main__':
    
        n = args.num_cores
        agents = [None] * n
        for a in range(0, n):
            agents[a] = lrn.create_agent(args.atari_env, args.t_max, args.T_max, args.C, shared)
        
        lock = Lock()
        processes = [None] * n
        for p in range(0, n):
            processes[p] = Process(target=agents[p].run, args=(lock, p)).start()

# IF EVALUATION mode -> evaluate a test run (rewards, video)
else:

    ag = lrn.create_agent_for_evaluation(args.file_name)
    ag.evaluate()    
