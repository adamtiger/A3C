from multiprocessing import Process, Lock, Pool
import argparse
import learner as lrn
import logger

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
parser.add_argument('--eval-num', type=int, default=10, metavar='N',
        help='the number of evaluations in an evaluation session (default:10)')
parser.add_argument('--train-mode', action='store_true',
        help='training or evaluation')
parser.add_argument('--file-name', default='model.json', metavar='S',
        help='the name of the file where the model was saved (default:model.json)')

args = parser.parse_args()

logger.create_folders(args.atari_env, args.num_cores, args.t_max, args.T_max, args.C)

# IF TRAIN mode -> train the learners

shared = lrn.create_shared(args.atari_env)

def executable(p):
    lrn.execute_agent(p, args.atari_env, args.t_max, args.T_max, args.C, args.eval_num, shared)

if (args.train_mode):
    
    # start the processes
    if __name__ == '__main__':
        
        n = args.num_cores
        l = Lock()
        
        pool = Pool(n, initializer = lrn.init_lock, initargs = (l,))
        idcs = [0] * n
        for p in range(0, n):
            idcs[p] = p
            
        pool.map(executable, idcs)
            
        pool.close()
        pool.join()
        
        shared.print_mtx()
        
        
# IF EVALUATION mode -> evaluate a test run (rewards, video)
else:

    ag = lrn.create_agent_for_evaluation(args.file_name)
    ag.evaluate()    
