import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from tqdm import tqdm

class PRM():
    def __init__(self, PRM_name, device, max_length=4096):
        self.model_dir = os.path.join('PRM',PRM_name)
        self.max_length = max_length
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'

        self.candidate_tokens = self.tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1] # 12902

    def covert_to_input(self, problem, thoughts):
        discard = 0
        while True:
            prm_thoughts = '\n'.join([f"Step {i+1+discard}: {thought} ки" for i, thought in enumerate(thoughts[discard:])])
            input_text = problem + ' ' + prm_thoughts
            input_for_prm = torch.tensor([self.tokenizer.encode(input_text)]).to(self.device)

            if input_for_prm.size(1) <= self.max_length or discard == len(thoughts)-1:
                break
            else:
                discard += 1

        return input_for_prm

    def get_step_scores(self, input_for_prm):
        n_tokens = input_for_prm.size(1)

        with torch.no_grad():
            logits = self.model(input_for_prm).logits[:,:,self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0] 
            step_scores = scores[input_for_prm == self.step_tag_id].cpu().tolist()

        return step_scores,n_tokens

# Test
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    PRM_name = 'MATH-Shepherd-Mistral-7B-PRM'
    reward_model = PRM(PRM_name, device)

    # question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
    # output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки""" # 18 is right
    # output2 = """Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки"""
    # output3 = """Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки"""
    # output4 = """Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * 2 = $18 every day at the farmers' market. ки""" # 18 is right
    # output4w = """Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * 2 = $17 every day at the farmers' market. ки"""
    # output0 = """Step 0: This is a difficult problem. ки"""
    # outputs = '\n'.join([output1, output2, output3, output4, output4w, output0])

    # for output in [output1, output2, output3, output4, output4w, output0, outputs]:
    #     step_scores = reward_model.get_step_scores(question, output)
    #     print(step_scores)

    # question = """Tim wants to invest some money in a bank which compounds quarterly with an annual interest rate of $7\%$. To the nearest dollar, how much money should he invest if he wants a total of $\$60,\!000$ at the end of $5$ years?"""
    question = """What are all values of $p$ such that for every $q>0$, we have   $$\\frac{3(pq^2+p^2q+3q^2+3pq)}{p+q}>2p^2q?$$ Express your answer in interval notation in decimal form."""
    output1 = """
    In this step, we plan for the task and conduct reasoning as follows.
    To follow the reverse calculation plan, we need to use the compound interest formula to find the present value (the amount Tim should invest now). The formula for the future value \( A \) of an investment compounded \( n \) times per year at an annual interest rate \( r \) over \( t \) years is:

    \[ A = P \left(1 + \frac{r}{n}\right)^{nt} \]

    Where:
    - \( A \) is the future value (\$60,000),
    - \( P \) is the present value (the amount to be found),
    - \( r \) is the annual interest rate (7% or 0.07),
    - \( n \) is the number of times interest is compounded per year (quarterly, so \( n = 4 \)),
    - \( t \) is the time the money is invested for (5 years).

    We need to solve for \( P \). Rearranging the formula to solve for \( P \):

    \[ P = \frac{A}{\left(1 + \frac{r}{n}\right)^{nt}} \]

    Substituting the given values:

    \[ P = \frac{60000}{\left(1 + \frac{0.07}{4}\right)^{4 \times 5}} \]

    This is the next step in the reverse calculation process.
    """

    output2 = """
    In this step, we refine the previous step to enhance clarity and correctness.
    The reasoning provided is correct up to the point of setting up the formula and substituting the values. The next step is to correctly apply the formula and perform the calculation. Here is the refined and correct next step:

    \[ P = \frac{60000}{\left(1 + \frac{0.07}{4}\right)^{4 \times 5}} \]

    This simplifies to:

    \[ P = \frac{60000}{\left(1 + 0.0175\right)^{20}} \]

    \[ P = \frac{60000}{(1.0175)^{20}} \]

    This is the correct expression to find the present value \( P \). The next step would be to calculate the value of \( (1.0175)^{20} \) and then divide 60000 by that value to find \( P \).
    To find the value of \( P \), we need to calculate \( (1.0175)^{20} \) and then divide 60000 by that value.

    First, calculate \( (1.0175)^{20} \):

    \[ (1.0175)^{20} \approx 1.414778 \]

    Now, divide 60000 by this value:

    \[ P = \frac{60000}{1.414778} \approx 42407.41 \]

    Rounding to the nearest dollar, we get:

    \[ P \approx 42407 \]

    So, the amount Tim should invest is \(\boxed{42407}\).
    """

    input_for_prm = reward_model.covert_to_input(question, [output1, output2, output1, output2, output1, output2, output1, output2, output1, output2, output1, output2])

    for i in tqdm(range(100)):
        step_scores,n_tokens = reward_model.get_step_scores(input_for_prm)
        print(step_scores)
        print(n_tokens)