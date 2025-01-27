from thought import LLM_thought
from collections import defaultdict
from LLM import LLM_api
import networkx as nx
import re
import random
random.seed(0)
import pdb
def default_equal(a, b):
    return float(a == b)
class ENV():
    def __init__(self, max_width=10, max_depth=10, answer_type="Numerical", LLM_args = {}, equal = default_equal,
                 zero_shot_mode="IO", debug_verbose=False) -> None:
        self.thoughts = {}
        self.current_tid = 0
        self.max_width = max_width
        self.max_depth = max_depth
        self.depth_2_id = defaultdict(list)
        self.leaf_nodes = set()
        self.problem = None
        self.node_unexplored = []
        self.vis_graph = nx.DiGraph()
        self.LLM = LLM_api(**LLM_args)
        self.score = {}
        self.answer_type = answer_type
        self.equal = equal
        assert self.answer_type in ["Numerical", "Choice", "Text", "Boolean"]
        self.zero_shot_mode = zero_shot_mode
        self.n_step = 0
        self.debug = debug_verbose
        self.thought_each_step = []

    def reason_1_step(self):
        # reason 1 step
        assert self.current_tid == self.node_unexplored[-1]
        self.node_unexplored.pop(-1)
        self.leaf_nodes.remove(self.current_tid)

        n_thought = len(self.thoughts)
    
        current_thought = self.thoughts[self.current_tid]
        current_text = current_thought.get_thought()
        prompt = 'Here is a problem and several reasoning steps.\n' + current_text + "\nPlease reason exactly ONE more step based on the current step here, and DONOT reason too many steps at once."

        ## Call LLM and get a new thought
        if self.debug:
            print('\n\n\n\n\n[reason_1_step]\n===========prompt===========', prompt, '==========response=============', sep="\n")
        LLM_response = self.LLM.get_text(prompt)
        if self.debug:
            print(LLM_response, "\n==========================")

        thought_text = current_text + f"\nSTEP{self.n_step}: \nIn this step, we conduct 1 reasoning step as follows.\n" + LLM_response + "\n"
        new_thought = LLM_thought(
            tid = n_thought + 1,
            thought = thought_text,
            parent_id = [self.current_tid],
            child_id=[],
            depth = current_thought.get_depth() + 1
        )
        self.thought_each_step.append(f"In this step, we conduct 1 reasoning step as follows.\n" + LLM_response + "\n")
        self.thoughts[self.current_tid].add_child(n_thought + 1)
        
        self.vis_graph.add_edge(self.current_tid, n_thought + 1)
        self.thoughts[n_thought + 1] = new_thought
        self.score[n_thought + 1] = self.thought_2_state(n_thought + 1)

        self.current_tid = n_thought + 1
        self.depth_2_id[new_thought.get_depth()].append(n_thought + 1)
        self.leaf_nodes.add(n_thought + 1)     
        self.node_unexplored.append(n_thought + 1)  
        
    def reason_1_step_decompose(self,text_before_decompose, text_decompose, subtask_id, text_previous_subtask=""):
        # reason 1 step
        assert self.current_tid == self.node_unexplored[-1]
        self.node_unexplored.pop(-1)
        self.leaf_nodes.remove(self.current_tid)

        n_thought = len(self.thoughts)
    
        current_thought = self.thoughts[self.current_tid]
        current_text = current_thought.get_thought()
        if subtask_id == 1:
            prompt = 'Here is a problem and several reasoning steps.\n' + text_before_decompose + \
            "\nFor the next step, the task is decomposed into subtasks.\n" + f"Please conduct the following Subtask{subtask_id} to continue the reasoning.\n" + text_decompose + "\nDONOT conduct more detailed decomposition for the subtask.\n"
        elif subtask_id == 2:
            prompt = 'Here is a problem and several reasoning steps.\n' + text_before_decompose + \
            f"\nFor the next step, the task is decomposed into subtasks, here are reasonings in the first subtask.\n" + text_previous_subtask + f"\nPlease conduct the following Subtask{subtask_id} to continue the reasoning.\n" + text_decompose + "\nDONOT conduct more detailed decomposition for the subtask.\n"
            
        else:
            prompt = 'Here is a problem and several reasoning steps.\n' + text_before_decompose + \
            f"\nFor the next step, the task is decomposed into subtasks, here are reasonings in the first {subtask_id -1} subtasks.\n" + text_previous_subtask + f"\nPlease conduct the following Subtask{subtask_id} to continue the reasoning.\n" + text_decompose + "\nDONOT conduct more detailed decomposition for the subtask.\n"

        ## Call LLM and get a new thought
        if self.debug:
            print('\n\n\n\n\n[reason_1_step]\n===========prompt===========', prompt, '==========response=============', sep="\n")
        LLM_response = self.LLM.get_text(prompt)
        if self.debug:
            print(LLM_response, "\n==========================")

        thought_text = current_text + "\n" + LLM_response + "\n"
        new_thought = LLM_thought(
            tid = n_thought + 1,
            thought = thought_text,
            parent_id = [self.current_tid],
            child_id=[],
            depth = current_thought.get_depth() + 1
        )
        self.thoughts[self.current_tid].add_child(n_thought + 1)
        
        self.vis_graph.add_edge(self.current_tid, n_thought + 1)
        self.thoughts[n_thought + 1] = new_thought

        self.current_tid = n_thought + 1
        self.depth_2_id[new_thought.get_depth()].append(n_thought + 1)
        self.leaf_nodes.add(n_thought + 1)     
        self.node_unexplored.append(n_thought + 1)  

    def reason_1_step_debate(self,text_before_debate, text_plan):
        # reason 1 step
        assert self.current_tid == self.node_unexplored[-1]
        self.node_unexplored.pop(-1)
        self.leaf_nodes.remove(self.current_tid)

        n_thought = len(self.thoughts)
    
        current_thought = self.thoughts[self.current_tid]
        current_text = current_thought.get_thought()

        prompt = 'Here is a problem and several reasoning steps.\n' + text_before_debate + \
        "\nFor the next step, we have decide a most promising plan:\n" + text_plan + f"\nPlease reason **exactly one** more step according to the plan here, and DONOT reason too many steps at once."
        ## Call LLM and get a new thought
        if self.debug:
            print('\n\n\n\n\n[reason_1_step]\n===========prompt===========', prompt, '==========response=============', sep="\n")
        LLM_response = self.LLM.get_text(prompt)
        if self.debug:
            print(LLM_response, "\n==========================")

        thought_text = text_before_debate + f"\nSTEP{self.n_step}: \n" + f"In this step, we plan for the task and conduct reasoning as follows.\nPlan:\n" + text_plan + "\nReasoning:\n" + LLM_response + "\n"
        new_thought = LLM_thought(
            tid = n_thought + 1,
            thought = thought_text,
            parent_id = [self.current_tid],
            child_id=[],
            depth = current_thought.get_depth() + 1
        )
        self.thought_each_step.append(f"In this step, we plan for the task and conduct reasoning as follows.\nPlan:\n" + text_plan + "\nReasoning:\n" + LLM_response + "\n")
        self.thoughts[self.current_tid].add_child(n_thought + 1)
        self.vis_graph.add_edge(self.current_tid, n_thought + 1)
        self.thoughts[n_thought + 1] = new_thought
        self.score[n_thought + 1] = self.thought_2_state(n_thought + 1)

        self.current_tid = n_thought + 1
        self.depth_2_id[new_thought.get_depth()].append(n_thought + 1)
        self.leaf_nodes.add(n_thought + 1)     
        self.node_unexplored.append(n_thought + 1)  

    def refine_thought(self):
        # refine current thought
        assert self.current_tid == self.node_unexplored[-1]
        self.node_unexplored.pop(-1)

        thought = self.thoughts[self.current_tid]
        current_text = thought.get_thought()
        prompt = 'Here is a problem and several reasoning steps.\n' + current_text + "\nPlease check and refine the current thought here, and DONOT conduct further reasoning or calculation."
        ## Call LLM and get a new thought
        if self.debug:
            print('\n\n\n\n\n[refine_thought]\n===========prompt===========', prompt, '==========response=============', sep="\n")
        LLM_response = self.LLM.get_text(prompt)
        if self.debug:
            print(LLM_response, "\n==========================")

        
        thought.set_thought(current_text + f"\nSTEP{self.n_step}: \nIn this step, we refine the previous step to enhance clarity and correctness.\n" + LLM_response)
        self.thought_each_step.append(f"In this step, we refine the previous step to enhance clarity and correctness.\n" + LLM_response)

        self.node_unexplored.append(self.current_tid)
        self.score[self.current_tid] = self.thought_2_state(self.current_tid)
        self.vis_graph.add_edge(self.current_tid, self.current_tid)

    def state_transition(self, add_prefix = False):
        assert self.current_tid == self.node_unexplored[-1]
        self.node_unexplored.pop(-1)
        if len(self.node_unexplored) == 0:
            return -1
        if add_prefix:
            self.leaf_nodes.remove(self.current_tid)
        current_thought = self.thoughts[self.current_tid].get_thought()
        next_thought = self.thoughts[self.node_unexplored[-1]].get_thought()
        ## find the common prefix of the states before and after transition
        common_prefix = []
        for i in range(min(len(current_thought), len(next_thought))):
            if current_thought[i] == next_thought[i]:
                common_prefix.append(current_thought[i])
            else:
                break
        common_prefix = "".join(common_prefix)
       
        if len(common_prefix) > 11 and 'subtask' in common_prefix[-10:].lower():
            new_text = common_prefix + current_thought[len(common_prefix):] + "Subtask" + next_thought[len(common_prefix):]
            current_text = common_prefix + current_thought[len(common_prefix):]
        elif len(common_prefix) > 11 and 'plan' in common_prefix[-10:].lower():
            new_text = common_prefix + current_thought[len(common_prefix):] + "Plan" + next_thought[len(common_prefix):]
            current_text = common_prefix + current_thought[len(common_prefix):]
        else:
            new_text = common_prefix + current_thought[len(common_prefix):] + next_thought[len(common_prefix):]
            current_text = common_prefix + current_thought[len(common_prefix):]
        
        self.current_tid = self.node_unexplored[-1]
        if add_prefix:
            self.thoughts[self.current_tid].set_thought(new_text)
            return current_text
  
    def decompose(self):
        # decompose thought
        assert self.current_tid == self.node_unexplored[-1]
        self.node_unexplored.pop(-1)
        self.leaf_nodes.remove(self.current_tid)

        current_thought = self.thoughts[self.current_tid]
        current_text = current_thought.get_thought()
        prompt = 'Here is a problem and several reasoning steps.\n' + current_text + "\nPlease decompose the current task into subtasks, where we can solve the original problem by combining these results of subtasks.\nOnly provide subtasks decomposition here, and DONOT conduct specific reasoning or calculation.\nUse the format '### Subtask1: subtask1'."
        ## Call LLM and get a few new thought
        if self.debug:
            print('\n\n\n\n\n[decompose]\n===========prompt===========', prompt, '==========response=============', sep="\n")
        LLM_response = self.LLM.get_text(prompt)
        if self.debug:
            print(LLM_response, "\n==========================")
        if "###" not in LLM_response:
            LLM_response = [LLM_response]
        else:
            LLM_response = LLM_response.split("### ")
        
        LLM_response = [x for x in LLM_response if len(x) > 10]
        for i,r in enumerate(LLM_response):
            idx = len(r) - 1
            while r[idx] == "\n":
                idx -= 1
            LLM_response[i] = r[:idx+1]
        if len(LLM_response) == 0:
            LLM_response = ["There problem cannot be divided into subtask, please refer to the original problem."]
            print("Warning: no subtask proposed")

        new_thought_id = []
        for i, text in enumerate(LLM_response[::-1]):
            new_thought = LLM_thought(
                tid = len(self.thoughts) + 1,
                thought = current_text + "\n" + text,
                parent_id = [self.current_tid],
                child_id=[],
                depth = current_thought.get_depth() + 1
            )
            new_tid = len(self.thoughts) + 1
            new_thought_id.append(new_tid)
            self.thoughts[self.current_tid].add_child(new_tid)
            self.thoughts[new_tid] = new_thought

            self.depth_2_id[new_thought.get_depth()].append(new_tid)
            self.leaf_nodes.add(new_tid)
            self.node_unexplored.append(new_tid)
            self.vis_graph.add_edge(self.current_tid, new_tid)

        self.current_tid = self.node_unexplored[-1] 

        step_text = ""
        for i,text in enumerate(LLM_response):
            self.reason_1_step_decompose(text_decompose=text.strip('\n'), text_before_decompose=current_text, text_previous_subtask = step_text, subtask_id=i+1)
            
            if i != len(LLM_response) - 1:
                step_text = self.state_transition(add_prefix=True)
                step_text = step_text[len(current_text):]

        self.simplify_decompose(current_text)
        return 
    
    def simplify_decompose(self, text_before_decompose):
        current_thought = self.thoughts[self.current_tid].get_thought()
        current_thought = current_thought[len(text_before_decompose):]

        prompt = 'Here are a few detailed reasoning subtasks of a problem.\n' + current_thought + "\nPlease give a clear and concise summary of these subtasks, keeping the key reasoning and results in each subtask. \nOnly provide the summary here, and DONOT conduct more reasoning or calculation."
        ## Call LLM and get a new thought
        if self.debug:
            print('\n\n\n\n\n[simplify_decompose]\n===========prompt===========', prompt, '==========response=============', sep="\n")
        LLM_response = self.LLM.get_text(prompt)
        if self.debug:
            print(LLM_response, "\n==========================")
        simplified_text = text_before_decompose + f"\nSTEP{self.n_step}: \nIn this step, the task is decomposed into a few subtasks, and the following is a summary of the reasonings in these subtasks:\n" + LLM_response
        self.thought_each_step.append(f"In this step, the task is decomposed into a few subtasks, and the following is a summary of the reasonings in these subtasks:\n" + LLM_response)
        self.thoughts[self.current_tid].set_thought(simplified_text)
        score = self.thought_2_state(self.current_tid)
        self.score[self.current_tid] = score
    
    def debate(self):
        # debate thought
        assert self.current_tid == self.node_unexplored[-1]
        self.node_unexplored.pop(-1)
        self.leaf_nodes.remove(self.current_tid)

        current_thought = self.thoughts[self.current_tid]
        current_text = current_thought.get_thought()
        prompt = 'Here is a problem and several reasoning steps.\n' + current_text + "\nPlease propose three different alternative plans for solving the problem in the current step.\nOnly provide plans here, and DONOT conduct specific reasoning or calculation.\nUse the format '### Plan1: plan1'."
        ## Call LLM and get a few new thought
        if self.debug:
            print('\n\n\n\n\n[debate]\n===========prompt===========', prompt, '==========response=============', sep="\n")
        LLM_response = self.LLM.get_text(prompt)
        if self.debug:
            print(LLM_response, "\n==========================")
        if "###" not in LLM_response:
            LLM_response = [LLM_response]
        else:
            LLM_response = LLM_response.split("### ")

        LLM_response = [x for x in LLM_response if len(x) > 10]
        for i,r in enumerate(LLM_response):
            idx = len(r) - 1
            while r[idx] == "\n":
                idx -= 1
            LLM_response[i] = r[:idx+1]

      
        if len(LLM_response) == 0:
            LLM_response = ["There is no plan proposed, please refer to the original problem."]
            print("Warning: no plan proposed")
        new_thought_id = []
        for i, text in enumerate(LLM_response[::-1]):
            new_thought = LLM_thought(
                tid = len(self.thoughts) + 1,
                thought = current_text + "\n" + text,
                parent_id = [self.current_tid],
                child_id=[],
                depth = current_thought.get_depth() + 1
            )
            new_tid = len(self.thoughts) + 1
            new_thought_id.append(new_tid)
            self.thoughts[self.current_tid].add_child(new_tid)
            self.thoughts[new_tid] = new_thought
            self.depth_2_id[new_thought.get_depth()].append(new_tid)
            self.leaf_nodes.add(new_tid)
            self.node_unexplored.append(new_tid)
            self.vis_graph.add_edge(self.current_tid, new_tid)

        self.current_tid = self.node_unexplored[-1]

        self.aggregate_1(text_before_debate=current_text, plans=LLM_response, agg_nodes=new_thought_id)
    
    def aggregate_1(self, text_before_debate, plans, agg_nodes=[],):
        plan_text = ""
        for p in plans:
            plan_text = plan_text + p + "\n"

        prompt = 'Here is a problem and several reasoning steps.\n' + text_before_debate + \
         'Currently, we have several alternative plans for solving the problem in the current step.\n' + plan_text + "\n" + "Please review and compare these plans carefully, and tell which one is most promising for further reasoning. Only compare the plans here, and DONOT conduct further reasoning or calculation.\
Use the format \'The most promising plan is Plan[INDEX]: [REASON]\', where [INDEX] is an integer index of plan and [REASON] is detailed analyse.\n"
        ## Call LLM and get a new thought
        if self.debug:
            print('\n\n\n\n\n[aggregate_1]\n===========prompt===========', prompt, '==========response=============', sep="\n")
        LLM_response = self.LLM.get_text(prompt)
        ## find the most promising plan
        #format:  The most promising plan is Plan1: 
        try:
            match = re.search("The most promising plan is [Pp]lan([0-9]+):", LLM_response)
            if match is not None:
                plan_index = int(match.group(1))
                best_plan = plans[plan_index - 1]
                best_plan = re.sub("[Pp]lan[0-9]+: ", "", best_plan)
            else:
                plan_index = 1
                best_plan = plans[0]
                best_plan = re.sub("[Pp]lan[0-9]+: ", "", best_plan)
        except:
            print("Warning: no plan selected")
            plan_index = 1
            best_plan = plans[0]
            best_plan = re.sub("[Pp]lan[0-9]+: ", "", best_plan)

        if self.debug:
            print(LLM_response, "\n==========================")

        ## remove the idx of the plan
        
        thought_text = text_before_debate + f"\nSTEP{self.n_step}: \n" + "In the current step, we seek the most promising plan for the task as follows\n" +\
            best_plan + "\n"


        agg_depths = [self.thoughts[n].get_depth() for n in agg_nodes]
        new_thought = LLM_thought(
            tid = len(self.thoughts) + 1,
            thought = thought_text + "\n",
            parent_id = agg_nodes,
            child_id=[],
            depth = max(agg_depths) + 1
        )
        new_tid = len(self.thoughts) + 1

        for n in agg_nodes:
            self.thoughts[n].add_child(new_tid)
            self.vis_graph.add_edge(n, new_tid)
            self.leaf_nodes.remove(n)

        while self.node_unexplored[-1] in agg_nodes:
            self.node_unexplored.pop(-1)
            if len(self.node_unexplored) == 0:
                break

        self.depth_2_id[new_thought.get_depth()].append(new_tid)
        self.leaf_nodes.add(new_tid)
        self.thoughts[new_tid] = new_thought
        self.node_unexplored.append(new_tid)
        self.current_tid = new_tid

        self.reason_1_step_debate(text_before_debate=text_before_debate, text_plan=best_plan)


    def get_answer(self):
        ## get answer and end the problem
        prompt = 'Here is a problem and several reasoning steps.\n' 
        for n in self.leaf_nodes:
            current_thought = self.thoughts[n].get_thought()
            # find the longest common prefix of current thought and the prompt
            common_prefix = []
            for i in range(len(prompt)):
                if prompt[i] == current_thought[i]:
                    common_prefix.append(prompt[i])
                else:
                    break
            common_prefix = "".join(common_prefix)
            thought_part = current_thought[len(common_prefix):]
            
            prompt += thought_part + "\n"
        
        ## Call LLM and get a new thought
        if self.answer_type == "Numerical":
            prompt += "Please generate the of the answer for the problem. Please end the answer with \"The answer is numerical_answer\""
            # prompt += "solution: \n"
        elif self.answer_type == "Choice":
            prompt += "End the answer with \"The answer is (CHOICE)\""
        elif self.answer_type == "Text":
            prompt += "\nPlease generate the answer for the problem. Wrap the answer with \\boxed{{answer}}"
        elif self.answer_type == "Boolean":
            prompt += "Please generate the answer for the problem. In the end of your answer, conclude the answer with \"The answer is yes\" or \"The answer is no\""

        if self.debug:
            print('\n\n\n\n\n[get_answer]\n===========prompt===========', prompt, '==========response=============', sep="\n")
        LLM_response = self.LLM.get_text(prompt)
        if self.debug:
            print(LLM_response, "\n==========================")

        new_thought = LLM_thought(
            tid = len(self.thoughts) + 1,
            thought = thought_part + "\n" + LLM_response + "\n",
            parent_id = self.leaf_nodes,
            child_id=[],
            depth = self.thoughts[self.current_tid].get_depth() + 1
        )
        new_tid = len(self.thoughts) + 1
        self.thoughts[new_tid] = new_thought
        self.thought_each_step.append("In this step, we generate the answer for the problem as follows.\n" + LLM_response + "\n")
        self.score[new_tid] = self.thought_2_state(new_tid)
        self.thoughts[self.current_tid].add_child(new_tid)
        self.depth_2_id[new_thought.get_depth()].append(new_tid)
        self.leaf_nodes.add(new_tid)
        self.node_unexplored.append(new_tid)
        self.vis_graph.add_edge(self.current_tid, new_tid)
        self.current_tid = new_tid

        #### EXTRACT ANSWER
        if self.answer_type == "Numerical":
            ori_response = LLM_response
            location = LLM_response.lower().find("the answer is")
            try:
                if location != -1:
                    LLM_response = LLM_response.lower().split("the answer is")[-1]
                ## only keep 0-9 and .
                LLM_response = re.sub(r"[^0-9.]", "", LLM_response)
                if LLM_response[-1] == ".":
                    LLM_response = LLM_response[:-1]
                LLM_response = float(LLM_response)

            except:
                print(ori_response)
                LLM_response = float(-998244353)
            return LLM_response
        
        elif self.answer_type == "Choice":
            # template = "The correct answer is (C)"
            match_result = re.search("the answer is \([a-d]\)", LLM_response.lower())
            if match_result is None:
                match_result = re.search("the answer is [a-d]", LLM_response.lower())
            if match_result is None:
                print("Warning: answer is not a choice")
                print(LLM_response)
                choice = random.choice(["A", "B", "C", "D"])
                return choice
            else:
                choice = match_result.group(0)[-2].upper()

            return choice
        elif self.answer_type == "Text":
            return LLM_response
        elif self.answer_type == "Boolean":
            template = "the answer is yes"
            flg = ""
            if template in LLM_response.lower():
                flg = "yes"
            elif "the answer is no" in LLM_response.lower():
                flg = "no"
            else:
                print("Warning: answer is not a boolean")
                print(LLM_response)
                flg = random.choice(["yes", "no"])
            return flg
        
    def reset(self):
        # reset
        self.problem = None
        self.thoughts = {}
        self.current_tid = 0
        self.depth_2_id = defaultdict(list)
        self.thought_each_step = []
        self.leaf_nodes = set()
        self.node_unexplored = []
        self.vis_graph = nx.DiGraph()
        self.score = {}

        self.LLM.reset_token()

    def get_current_tid(self):
        return self.current_tid
    
    def visualize(self):
        # visualize the thoughts
        from matplotlib import pyplot as plt
        plt.figure(figsize=(10,10))
        # position of the nodes (x=depth, y=position in depth)
        pos = {k: (v.get_depth(), i) for i, (k, v) in enumerate(self.thoughts.items())}
        # bfs to set the y coordinate
        y_range = {0: (-3,3)}
        search_q = []
        search_q.append(0)
        searched = set()
        while search_q:
            node = search_q.pop(0)
            if node in searched:
                continue
            searched.add(node)
            if len(self.thoughts[node].get_child_id()) == 0:
                continue    
            elif len(self.thoughts[node].get_child_id()) == 1:
                search_q.append(self.thoughts[node].get_child_id()[0])
                y_range[self.thoughts[node].get_child_id()[0]] = (y_range[node][0], y_range[node][1])
            else:
                for i, child in enumerate(self.thoughts[node].get_child_id()):
                    search_q.append(child)
                    width = y_range[node][1] - y_range[node][0]
                    y_range[child] = (y_range[node][0] + i * width / len(self.thoughts[node].get_child_id()),\
                                       y_range[node][0] + (i + 1) * width / len(self.thoughts[node].get_child_id()))
                    

        pos = {k: (v.get_depth(), (y_range[k][0] + y_range[k][1]) / 2) for k, v in self.thoughts.items()}

        nx.draw(self.vis_graph, pos, with_labels=True, font_size=20, node_size=500)
        plt.savefig("../result/thoughts.png")
        plt.close()

    def set_problem(self, problem, ans, choices=None):
        # set problem
        self.reset()
        if self.answer_type == "Numerical":
            self.ans = float(ans)
            self.problem = "PROBLEM: " + problem
        elif self.answer_type == "Choice":
            self.ans = ans
            self.problem = "PROBLEM: " + problem + "\nChoices:\n" + choices
            self.choices = choices
        elif self.answer_type == "Text":
            self.ans = ans
            self.problem = "PROBLEM: " + problem
        elif self.answer_type == "Boolean":
            self.ans = ans
            self.problem = "PROBLEM: " + problem
        ## thought 0 is the problem
        thought = LLM_thought(
            tid = 0,
            thought = self.problem,
            parent_id = [],
            child_id = [],
            depth = 0
        )
        self.thoughts[0] = thought
        self.leaf_nodes.add(0)
        self.node_unexplored.append(0)
        self.current_tid = 0
        self.score[0] = self.thought_2_state(0)
        self.n_step = 0

    def check_width_depth(self):
        # check width and depth
        for k, v in self.depth_2_id.items():
            if len(v) > self.max_width:
                return False
        if len(self.depth_2_id) > self.max_depth:
            return False
        return True

    def thought_2_state(self, thought_id=None, short=False):
        def extract_state(res_string):
            ## a post process for extracting the score. there are 8 of them,
            state_key = ["A1", "A2", "A3", "B1", "B2", "C1", "C2"]
            state = {}
            for key in state_key:
                match = re.search(f"{key}[:=\[a-zA-Z \n]*(0|1|2|3)", res_string)
                if match:
                    score = int(match.group(1))
                else:
                    score = 0
                state[key] = score
            return state
        # thought to state
        template = '''
Please evaluate the current step from the following aspects. 
A) Correctness
    A1: Correctness of modeling:
    Whether the current step is correctly derived from the origin problem.
    A2: Clarity for further reasoning:
    Whether the current step is clearly presented, without ambiguity, to support further reasoning.
    A3: Correctness of calculation:
    Whether the numerical computation in the current step is performed correctly. 
B) Complexity
    B1: Complexity to reach the final answer:
    Whether there still requires complex reasoning or calculation to reach the final answer from the current step.
    B2: Alternative methods in further reasoning:
    Whether there exist multiple alternative methods to solve the problem in the current step.
C) Completeness
    C1: Closeness to the final solution:
    Whether the current step is close enough to directly reach the final answer.
    C2: Completeness within the step:
    Whether all necessary elements within this specific step are known from the problem or previous steps.
For each aspect, please score 1 for False, 2 for Unsure, 3 for True, and socre 0 if the current step does not involve this aspect. Please attach reason for each score.
Use the format 'A1 score=[SCORE] reason=[REASON]'.
Only score the current reasoning step here, and DONOT conduct further reasonings.
'''
        if thought_id is None:
            thought_id = self.current_tid

        prompt = 'Here is a problem and several reasoning steps.\n' + self.thoughts[thought_id].get_thought() + template
    
        ## Call LLM and get a new thought
        if self.debug:
            print('\n\n\n\n\n[thought_2_state]\n===========prompt===========', prompt, '==========response=============', sep="\n")
        LLM_response = self.LLM.get_text(prompt)
        if self.debug:
            print(LLM_response, "\n==========================")
        state = extract_state(LLM_response)
        return state
    
    def check_finished(self):
        template = "\nPlease check whether there already exists a final answer for the entire problem in the current step. Just return \'yes\' or \'no\'."
        prompt = 'Here is a problem and several reasoning steps.\n' + self.thoughts[self.current_tid].get_thought() + template
        ## Call LLM and get a new thought
        if self.debug:
            print('\n\n\n\n\n[check_finished]\n===========prompt===========', prompt, '==========response=============', sep="\n")
        LLM_response = self.LLM.get_text(prompt)
        if self.debug:
            print(LLM_response, "\n==========================")
        if "yes" in LLM_response.lower():
            return True
        else:
            return False
    
    def print_childs(self):
        for k, v in self.thoughts.items():
            print(f"tid: {k}, child: {v.get_child_id()}")

    def print_score(self):
        for k, v in self.score.items():
            print(f"tid: {k}, score: {v}")

    def print_token_usage(self):
        self.LLM.print_usage()

    def step(self, action):
        if self.n_step == 0 and action == "Rf":
            action = "R"
        self.n_step += 1
        if self.n_step > self.max_depth:
            ans = self.get_answer()
            return self.score[self.current_tid], self.equal(ans, self.ans), True
        # api for RL
        if action == "R":
            self.reason_1_step()
        elif action == "D":
            self.decompose()
            # self.aggregate_1(prompt_type=0)
        elif action == "Rf":
            self.refine_thought()
        elif action == "Db":
            self.debate()
        elif action == "Ga":
            ans = self.get_answer()
            return self.score[self.current_tid], self.equal(ans, self.ans), True
        else:
            print("Invalid action")
            return

        finished = self.check_finished()
        if finished:
            ans = self.get_answer()
            return self.score[self.current_tid], self.equal(ans, self.ans), True
        
        return self.score[self.current_tid], 0, False

    def store_state(self):
        store_dict = {
            "thoughts": self.thoughts,
            "current_tid": self.current_tid,
            "max_width": self.max_width,
            "max_depth": self.max_depth,
            "depth_2_id": self.depth_2_id,
            "leaf_nodes": self.leaf_nodes,
            "problem": self.problem,
            "node_unexplored": self.node_unexplored,
            "vis_graph": self.vis_graph,
            "ans": self.ans,
            "score": self.score
        }
        return store_dict

    def recover_state(self, store_dict):
        self.thoughts = store_dict["thoughts"]
        self.current_tid = store_dict["current_tid"]
        self.max_width = store_dict["max_width"]
        self.max_depth = store_dict["max_depth"]
        self.depth_2_id = store_dict["depth_2_id"]
        self.leaf_nodes = store_dict["leaf_nodes"]
        self.problem = store_dict["problem"]
        self.node_unexplored = store_dict["node_unexplored"]
        self.vis_graph = store_dict["vis_graph"]
        self.ans = store_dict["ans"]
        self.score = store_dict["score"]
