from environment_simple import ENV
from modelscope.msdatasets import MsDataset
from math_evaluate import is_equiv

import random
import numpy as np
import pandas as pd
import json
import os

class RLEnv():
    def __init__(self,dataset,is_test,LLM_name,problem_indexs,max_depth,max_width,random_problems,random_seed):
        super(RLEnv, self).__init__()
        self.action_space = {0:"R", 1:"D", 2:"Db", 3:"Rf", 4:"Ga"}
        self.observation_space = {0:"A1", 1:"A2", 2:"A3", 3:"B1", 4: "B2", 5: "C1", 6:"C2"}

        self.dataset = dataset
        self.is_test = is_test
        self.LLM_args = {"model":LLM_name}
        self.max_depth = max_depth
        self.max_width = max_width

        if self.dataset == 'MATH':
            self.ds =  MsDataset.load('modelscope/competition_math', cache_dir='data', subset_name='default', split=('test' if is_test else 'train'))
            self.answer_type = 'Text'
            self.total_problems = len(self.ds)
            self.core = ENV(answer_type=self.answer_type, LLM_args=self.LLM_args, max_depth=self.max_depth, max_width=self.max_width, equal = is_equiv, debug_verbose=True)
        
        elif self.dataset == 'GSM8K':
            self.ds =  MsDataset.load('modelscope/gsm8k', cache_dir='data', subset_name='main', split=('test' if is_test else 'train'))
            self.answer_type = 'Numerical'
            self.total_problems = len(self.ds)
            self.core = ENV(answer_type=self.answer_type, LLM_args=self.LLM_args, max_depth=self.max_depth, max_width=self.max_width, debug_verbose=True)
        
        elif self.dataset == 'GPQA':
            self.ds =  pd.read_csv(os.path.join('data', 'gpqa_main.csv'))
            self.answer_type = 'Choice'
            self.total_problems = len(self.ds)
            self.core = ENV(answer_type=self.answer_type, LLM_args=self.LLM_args, max_depth=self.max_depth, max_width=self.max_width, debug_verbose=True)
            
        elif self.dataset == 'MMLU-STEM':
            self.ds =  MsDataset.load('TIGER-Lab/MMLU-STEM', cache_dir='data', split=('test' if is_test else 'train'))
            self.answer_type = 'Choice'
            self.total_problems = len(self.ds)
            self.core = ENV(answer_type=self.answer_type, LLM_args=self.LLM_args, max_depth=self.max_depth, max_width=self.max_width, debug_verbose=True)

        elif self.dataset == 'StrategyQA':
            split=('test' if is_test else 'train')
            self.ds =  json.load(open(os.path.join('data', f'strategyQA_{split}.json'), "r"))
            self.answer_type = 'Boolean'
            self.total_problems = len(self.ds)
            self.core = ENV(answer_type=self.answer_type, LLM_args=self.LLM_args, max_depth=self.max_depth, max_width=self.max_width, debug_verbose=True)

        else:
            raise ValueError("Dataset not supported")
        
        if problem_indexs is None:
            self.problem_indexs = list(range(self.total_problems))
        else:
            self.problem_indexs = problem_indexs
        self.num_problems = len(self.problem_indexs)

        assert np.max(self.problem_indexs) <= self.total_problems, "num_problems should be less than total_problems"
        self.current_problem = 0

        self.random_problems = random_problems
        random.seed(random_seed)

        self.finished = False

    def reset(self,true_reset=True):
        current_index = self.problem_indexs[self.current_problem]

        if self.dataset == 'MATH':
            self.problem = self.ds[current_index]['problem']
            self.ans = self.ds[current_index]['solution']

        elif self.dataset == 'GSM8K':
            self.problem = self.ds[current_index]['question']
            tmp_ans = self.ds[current_index]['answer']
            self.ans = tmp_ans.split("####")[1].replace(" ", "").replace("\n", "").replace(",", "")

        elif self.dataset == 'GPQA':
            self.problem = self.ds.iloc[current_index]['Question']

            options = [self.ds.iloc[current_index]['Correct Answer'], self.ds.iloc[current_index]['Incorrect Answer 1'], self.ds.iloc[current_index]['Incorrect Answer 2'], self.ds.iloc[current_index]['Incorrect Answer 3']]
            random.shuffle(options)
            self.choices = ""
            for j in range(len(options)):
                self.choices += "(" + chr(ord('A') + j) + ") " + options[j] + "\n"

            tmp_ans = options.index(self.ds.iloc[current_index]['Correct Answer'])
            self.ans = chr(ord('A') + tmp_ans)

        elif self.dataset == 'MMLU-STEM':
            self.problem = self.ds[current_index]['question']
            
            self.choices = ""
            for i, c in enumerate(self.ds[current_index]["choices"]):
                self.choices += f"(" + chr(ord('A') + i) + ")" + c + "\n"
            
            tmp_ans = self.ds[current_index]['answer']
            self.ans = chr(ord('A') + tmp_ans)
        
        elif self.dataset == 'StrategyQA':
            self.problem = self.ds[current_index]['question']
            self.ans = "yes" if self.ds[current_index]['answer'] else "no"

        if self.random_problems:
            self.current_problem = random.randint(0,self.num_problems-1)
        else:
            self.current_problem = (self.current_problem + 1) % self.num_problems

        if self.current_problem == 0:
            self.finished = True

        if true_reset:
            self.core.reset()

            if self.answer_type == "Choice":
                self.core.set_problem(self.problem, self.ans, self.choices)
            else:
                self.core.set_problem(self.problem, self.ans)

            observation = self.core.score[self.core.current_tid]
            state = list(observation.values())

            return state, self.finished
        
        else:
            return None, self.finished

    def step(self, action):
        observation, reward, done=self.core.step(self.action_space[action])
        state = list(observation.values())

        return state, reward, done
    
if __name__ == "__main__":
    env = RLEnv(dataset='StrategyQA', is_test=True, LLM_name='Pro/Qwen/Qwen2.5-7B-Instruct', problem_indexs=None, max_depth=5, max_width=5, random_problems=False, random_seed=0)
    print(env.total_problems)
    for i in range(1):
        state, finished = env.reset()
        print(state)
        print(env.problem)
        # print(env.choices)
        print(env.ans)

        s,r,d = env.step(1)
        print(s,r,d)

        s,r,d = env.step(0)
        print(s,r,d)

        s,r,d = env.step(2)
        print(s,r,d)

        s,r,d = env.step(3)
        print(s,r,d)

        s,r,d = env.step(4)
        print(s,r,d)
        