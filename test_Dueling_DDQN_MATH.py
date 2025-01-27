import setproctitle
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append('env')

import os
import pickle
import torch
import time
from tqdm import tqdm

from Dueling_DQN_net import Dueling_DQN
from env.RL_env import RLEnv

class test_Dueling_DDQN_agent():
    def __init__(self,
                 manual_seed,
                 test_dataset,
                 test_LLM_name,
                 problem_indexs,
                 max_depth,
                 max_width,
                 train_dataset,
                 train_LLM_name,
                 model_dir,
                 model_name,
                 model_index,
                 save_dir,
                 ):

        print('Initializing...')

        # generate config
        config_data=locals()
        del config_data['self']
        time_data=time.strftime('%Y-%m-%d_%H-%M', time.localtime())
        config_data['code_dir']=os.path.split(os.path.realpath(__file__))[0]
        config_data['time']=time_data

        # random seed
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        self.test_dataset = test_dataset
        self.test_LLM_name = test_LLM_name
        self.problem_indexs= problem_indexs
        self.max_depth = max_depth
        self.max_width = max_width
        self.random_seed = manual_seed
        self.env = RLEnv(dataset=self.test_dataset, is_test=True, LLM_name=self.test_LLM_name, problem_indexs=self.problem_indexs, max_depth=self.max_depth, max_width=self.max_width, random_problems=False, random_seed=self.random_seed)
        self.n_actions = len(self.env.action_space)
        self.n_features = len(self.env.observation_space)

        self.train_dataset = train_dataset
        self.train_LLM_name = train_LLM_name
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_folder = os.path.join(self.model_dir, self.train_LLM_name.split("/")[-1], self.train_dataset, self.model_name)
        self.model_index = model_index
        
        self.Dueling_DQN = Dueling_DQN(input_size=self.n_features,output_size=self.n_actions).to(self.device).eval()
        self.Dueling_DQN.load_state_dict(torch.load(os.path.join(self.model_folder, f"model_episode{model_index}.pth")))

        self.save_dir = save_dir
        self.save_folder = os.path.join(self.save_dir, 'train-'+self.train_dataset, 'train-'+self.train_LLM_name.split("/")[-1], 'test-'+self.test_dataset, 'test-'+self.test_LLM_name.split("/")[-1], f'{model_name}_index{model_index}_time'+time_data)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        ref_results = sorted(os.listdir(os.path.join(self.save_dir, 'train-'+self.train_dataset, 'train-'+self.train_LLM_name.split("/")[-1], 'test-'+self.test_dataset, 'test-'+self.test_LLM_name.split("/")[-1])))
        if len(ref_results) != 0:
            self.ref_folder = os.path.join(self.save_dir, 'train-'+self.train_dataset, 'train-'+self.train_LLM_name.split("/")[-1], 'test-'+self.test_dataset, 'test-'+self.test_LLM_name.split("/")[-1], ref_results[0])
        else:
            self.ref_folder = None

        print(self.test_LLM_name + ' on ' + self.test_dataset + ' initialized.')
    
    def choose_action(self,state):
        with torch.no_grad():
            state = torch.tensor(state,dtype=torch.float32).to(self.device)
            state = state.unsqueeze(0)
            actions_value = self.Dueling_DQN(state)
            action = torch.argmax(actions_value).item()

        return action
    
    def test(self):
        correct_list = list()
        for problem_index in tqdm(self.problem_indexs):

            if self.ref_folder is not None:
                if os.path.exists(os.path.join(self.ref_folder, f'thoughts_log_problem{problem_index}.txt')) and os.path.exists(os.path.join(self.ref_folder, f'record_problem{problem_index}.pkl')):
                    os.system(f"cp {os.path.join(self.ref_folder, f'record_problem{problem_index}.pkl')} {os.path.join(self.save_folder, f'record_problem{problem_index}.pkl')}")
                    os.system(f"cp {os.path.join(self.ref_folder, f'thoughts_log_problem{problem_index}.txt')} {os.path.join(self.save_folder, f'thoughts_log_problem{problem_index}.txt')}")
                    _, finished = self.env.reset(true_reset=False)
                    continue

            sys.stdout = open(os.path.join(self.save_folder,f'thoughts_log_problem{problem_index}.txt'),'w')

            assert problem_index == self.env.problem_indexs[self.env.current_problem]
            state, finished = self.env.reset()

            episode_problem = (self.env.problem, self.env.ans)
            episode_states = list()
            episode_actions = list()

            start_flag = True

            while True:
                episode_states.append(state)

                action = self.choose_action(state)
                if start_flag and action == 3:
                    action = 0
                start_flag = False

                state_next,reward_ORM,done = self.env.step(action)
                
                if done:
                    episode_actions.append(4)

                    if reward_ORM == 1:
                        correct_list.append(problem_index)

                    q_token, a_token = self.env.core.LLM.get_token()

                    sys.stdout.flush()
                    pickle.dump((episode_problem, episode_states, episode_actions, reward_ORM, q_token, a_token), open(os.path.join(self.save_folder, f"record_problem{problem_index}.pkl"), "wb"))
                    break

                episode_actions.append(action)
                state = state_next
                
        
        assert finished


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_seed', type=int, default=0, help='random seed')
    parser.add_argument('--test_dataset', type=str, default='MATH', help='dataset to test')
    parser.add_argument('--test_LLM_name', type=str, default='Qwen/Qwen2.5-14B-Instruct', help='LLM to test')
    parser.add_argument('--max_depth', type=int, default=5, help='maximum depth of the tree')
    parser.add_argument('--max_width', type=int, default=5, help='maximum width of the tree')
    parser.add_argument('--train_dataset', type=str, default='MATH', help='dataset to train')
    parser.add_argument('--train_LLM_name', type=str, default='Qwen/Qwen2.5-14B-Instruct', help='LLM to train')
    parser.add_argument('--model_dir', type=str, default='model', help='folder to load the model')
    parser.add_argument('--model_name', type=str, default='Dueling_DDQN_2025-01-17_02-48-13', help='name of the model')
    parser.add_argument('--model_index', type=int, default=3000, help='index of the model')
    parser.add_argument('--save_dir', type=str, default='test', help='folder to save the test results')
    args = parser.parse_args()


    if args.test_dataset == 'GSM8K':
        problem_indexs = list(range(1319))
    elif args.test_dataset == 'MATH':
        problem_indexs = list(range(5000))
    elif args.test_dataset == 'GPQA':
        problem_indexs = list(range(448))
    elif args.test_dataset == 'MMLU-STEM':
        problem_indexs = list(range(3153))
    elif args.test_dataset == 'StrategyQA':
        problem_indexs = list(range(687))


    # agent = test_Dueling_DDQN_agent(manual_seed=args.manual_seed, test_dataset=args.test_dataset, test_LLM_name=args.test_LLM_name, problem_indexs=problem_indexs, max_depth=args.max_depth, max_width=args.max_width, train_dataset=args.train_dataset, train_LLM_name=args.train_LLM_name, model_dir=args.model_dir, model_name=args.model_name, model_index=args.model_index, save_dir=args.save_dir)
    # correct_list = agent.test()
    # print(correct_list)


    import multiprocessing
    import numpy as np

    def test_partition(problem_index_partition):
        agent = test_Dueling_DDQN_agent(manual_seed=args.manual_seed, test_dataset=args.test_dataset, test_LLM_name=args.test_LLM_name, problem_indexs=problem_index_partition, max_depth=args.max_depth, max_width=args.max_width, train_dataset=args.train_dataset, train_LLM_name=args.train_LLM_name, model_dir=args.model_dir, model_name=args.model_name, model_index=args.model_index, save_dir=args.save_dir)
        agent.test()

        return True
    
    MAX_THREADS=10
    p = multiprocessing.Pool(MAX_THREADS)
    result=list()

    np.random.shuffle(problem_indexs)
    partition_size = len(problem_indexs)//MAX_THREADS
    for i in range(MAX_THREADS):
        if i == MAX_THREADS-1:
            result.append(p.apply_async(test_partition,args=(sorted(problem_indexs[i*partition_size:]),)))
        else:
            result.append(p.apply_async(test_partition,args=(sorted(problem_indexs[i*partition_size:(i+1)*partition_size]),)))

    for obj in tqdm(result):
        obj.get()
    
    p.close()
