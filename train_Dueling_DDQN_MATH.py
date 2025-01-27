import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append('env')

import os
import pickle
import torch
import json
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from PRM import PRM
from Dueling_DQN_net import Dueling_DQN
from DQN_buffer import Experience,ExperienceReplay
from env.RL_env import RLEnv

class Dueling_DDQN_agent():
    def __init__(self,
                 manual_seed,
                 dataset,
                 LLM_name,
                 problem_indexs_name,
                 random_problems,
                 PRM_name,
                 max_depth,
                 max_width,
                 save_interval,
                 save_dir,
                 learning_rate,
                 learning_rate_decay,
                 learning_rate_decay_interval,
                 gamma,
                 num_episodes,
                 batch_size,
                 start_epsilon,
                 min_epsilon,
                 epsilon_decay,
                 buffer_size,
                 target_update_interval
                 ):

        print('Initializing...')

        # generate config
        config_data=locals()
        del config_data['self']
        time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        config_data['code_dir']=os.path.split(os.path.realpath(__file__))[0]
        config_data['time']=time_data

        # random seed
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.manual_seed=manual_seed
        torch.manual_seed(self.manual_seed)
        if self.device=='cuda':
            torch.cuda.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)

        self.dataset = dataset
        self.LLM_name = LLM_name
        # self.problem_indexs = [8]
        # self.problem_indexs = [6, 8, 9, 11, 38, 49, 60, 74, 93, 109]
        self.problem_indexs = pickle.load(open(os.path.join('data', problem_indexs_name, LLM_name.split("/")[-1], dataset, 'indexs.pkl'), "rb"))
        self.random_problems = random_problems
        self.max_depth = max_depth
        self.max_width = max_width
        self.env = RLEnv(dataset=self.dataset, is_test=False, LLM_name=self.LLM_name, problem_indexs=self.problem_indexs, max_depth=self.max_depth, max_width=self.max_width, random_problems=self.random_problems, random_seed=self.manual_seed)
        self.n_actions = len(self.env.action_space)
        self.n_features = len(self.env.observation_space)

        self.save_dir = save_dir
        self.save_interval = save_interval
        self.save_folder = os.path.join(self.save_dir, self.LLM_name.split("/")[-1], self.dataset, 'Dueling_DDQN_'+time_data)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        with open(os.path.join(self.save_folder,'config.json'),'w') as f:
            json.dump(config_data,f)
        os.system(f'cp -r '+'env'+' '+os.path.join(self.save_folder,'env'))
        os.system(f'cp '+'train_Dueling_DDQN_MATH.py'+' '+os.path.join(self.save_folder,'train_Dueling_DDQN_MATH.py'))
        os.system(f'cp '+'Dueling_DQN_net.py'+' '+os.path.join(self.save_folder,'Dueling_DQN_net.py'))
        os.system(f'cp '+'DQN_buffer.py'+' '+os.path.join(self.save_folder,'DQN_buffer.py'))
        os.system(f'cp '+'PRM.py'+' '+os.path.join(self.save_folder,'PRM.py'))

        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_interval = learning_rate_decay_interval
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.target_update_interval = target_update_interval

        self.PRM_name = PRM_name
        self.PRM = PRM(PRM_name=self.PRM_name, device=self.device)
        
        self.Dueling_DQN = Dueling_DQN(input_size=self.n_features,output_size=self.n_actions).to(self.device)
        self.Dueling_DQN_target = copy.deepcopy(self.Dueling_DQN).to(self.device)
        self.optimizer = torch.optim.Adam(self.Dueling_DQN.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.learning_rate_decay_interval, gamma=self.learning_rate_decay)
        self.loss = torch.nn.MSELoss()
        self.memory = ExperienceReplay(capacity=self.buffer_size, random_seed=self.manual_seed)

        self.rewards = list()
        self.learn_count = 0

        print(self.LLM_name + ' on ' + self.dataset + ' initialized.')
        sys.stdout = open(os.path.join(self.save_folder,'thoughts_log.txt'),'a')
    
    def choose_action(self,state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0,self.n_actions)
        else:
            state = torch.tensor(state,dtype=torch.float32).to(self.device)
            state = state.unsqueeze(0)
            actions_value = self.Dueling_DQN(state)
            action = torch.argmax(actions_value).item()

        return action
        
    def learn(self):
        batch_memory = self.memory.sample(self.batch_size)
        states, actions, rewards, dones, next_states = batch_memory

        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones,dtype=torch.bool).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)

        action_values = self.Dueling_DQN(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.Dueling_DQN(next_states).max(1)[1]
            next_action_values = self.Dueling_DQN_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_action_values[dones] = 0

        target_values = rewards + self.gamma * next_action_values
        loss = self.loss(action_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_count += 1

        sys.stdout.flush()
        sys.stdout = sys.__stdout__
        tqdm.write(f"Loss={loss.item()}")
        
        if self.learn_count % self.target_update_interval == 0:
            self.Dueling_DQN_target.load_state_dict(self.Dueling_DQN.state_dict())
            tqdm.write("Target network updated.")

        sys.stdout = open(os.path.join(self.save_folder,'thoughts_log.txt'),'a')

    def save(self,episode_count):
        torch.save(self.Dueling_DQN.state_dict(), os.path.join(self.save_folder, f"model_episode{episode_count}.pth"))

        plt.figure(figsize=(8,6))
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.xlim(0, self.num_episodes)
        plt.savefig(os.path.join(self.save_folder, "rewards.png"))
        plt.close()

        plt.figure(figsize=(8,6))
        plt.plot([np.mean(self.rewards[max(0,i-100):i]) for i in range(1,len(self.rewards))])
        plt.xlabel('Episode')
        plt.ylabel('Smooth reward')
        plt.xlim(0, self.num_episodes)
        plt.savefig(os.path.join(self.save_folder, "smooth_rewards.png"))
        plt.close()

        with open(os.path.join(self.save_folder, "rewards.pkl"), "wb") as f:
            pickle.dump(self.rewards, f)
    
    def train(self):
        for episode_count in tqdm(range(self.num_episodes)):
            print('\n\n\n')
            print('++++++++++++++++++++++++++++++++')
            print(f'++++++++Episode {episode_count}++++++++')
            print('++++++++++++++++++++++++++++++++')
            state, _ = self.env.reset()

            episode_problem = (self.env.problem, self.env.ans)
            episode_states = list()
            episode_actions = list()
            episode_rewards = list()

            step_count = 0
            start_flag = True

            while True:
                episode_states.append(state)

                step_count += 1
                self.epsilon = max(self.min_epsilon,self.epsilon*self.epsilon_decay)

                action = self.choose_action(state)
                if start_flag and action == 3:
                    action = 0
                start_flag = False
                
                state_next,reward_ORM,done = self.env.step(action)

                if done:
                    reward = reward_ORM
                else:
                    thoughts = self.env.core.thought_each_step
                    input_for_prm = self.PRM.covert_to_input(self.env.problem,thoughts)
                    rewards,n_token = self.PRM.get_step_scores(input_for_prm)
                    reward = rewards[-1]

                    sys.stdout.flush()
                    sys.stdout = sys.__stdout__
                    tqdm.write(f"{n_token}")
                    sys.stdout = open(os.path.join(self.save_folder,'thoughts_log.txt'),'a')

                exp = Experience(state,action,reward,done,state_next)
                self.memory.append(exp)

                episode_rewards.append(reward)

                if len(self.memory) >= self.buffer_size:
                    self.learn()
                
                if done:
                    episode_actions.append(4)

                    episode_reward = np.mean(episode_rewards)
                    self.rewards.append(episode_reward)

                    sys.stdout.flush()
                    sys.stdout = sys.__stdout__
                    tqdm.write(f"Episode {episode_count} | Step {step_count}\tReward {episode_reward}\tAvgReward {np.mean(self.rewards[max(0,episode_count+1-100):episode_count+1])}\t{len(self.memory)}")
                    sys.stdout = open(os.path.join(self.save_folder,'thoughts_log.txt'),'a')

                    pickle.dump((episode_problem, episode_states, episode_actions, episode_rewards), open(os.path.join(self.save_folder, f"record_episode{episode_count}.pkl"), "wb"))
                    np.save(os.path.join(self.save_folder, f"rewards.npy"), np.array(self.rewards))

                    break
            
                episode_actions.append(action)
                state = state_next

            if (episode_count + 1) % self.save_interval == 0:
                self.save(episode_count + 1)

            self.lr_scheduler.step()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_seed', type=int, default=1, help='manual seed for reproducibility')
    parser.add_argument('--dataset', type=str, default='GPQA', help='dataset to use')
    parser.add_argument('--LLM_name', type=str, default='Qwen/Qwen2.5-14B-Instruct', help='name of the LLM')
    parser.add_argument('--problem_indexs_name', type=str, default='problem_indexs', help='folder to load the problem indexs')
    parser.add_argument('--random_problems', type=bool, default=True, help='whether to use random problems')
    parser.add_argument('--PRM_name', type=str, default='MATH-Shepherd-Mistral-7B-PRM', help='name of the PRM')
    parser.add_argument('--max_depth', type=int, default=5, help='maximum depth of the tree')
    parser.add_argument('--max_width', type=int, default=5, help='maximum width of the tree')
    parser.add_argument('--save_interval', type=int, default=10, help='interval to save the model')
    parser.add_argument('--save_dir', type=str, default='model', help='folder to save the model')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('--learning_rate_decay', type=float, default=0.5, help='decay rate of learning rate')
    parser.add_argument('--learning_rate_decay_interval', type=int, default=1000, help='interval to decay the learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
    parser.add_argument('--num_episodes', type=int, default=3000, help='number of episodes to train')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--start_epsilon', type=float, default=1.0, help='start epsilon for epsilon-greedy')
    parser.add_argument('--min_epsilon', type=float, default=0.0, help='minimum epsilon for epsilon-greedy')
    parser.add_argument('--epsilon_decay', type=float, default=0.9995, help='decay rate of epsilon')
    parser.add_argument('--buffer_size', type=int, default=500, help='size of the replay buffer')
    parser.add_argument('--target_update_interval', type=int, default=50, help='interval to update the target')
    args = parser.parse_args()

    agent = Dueling_DDQN_agent(**vars(args))
    agent.train()