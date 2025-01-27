import requests
import time
import httpx
from openai import OpenAI,AzureOpenAI
from API_key import api_key_dict

class OverloadError(Exception):
    pass

class LLM_api():
    def __init__(self, 
                 content = "", 
                 model = "Qwen/Qwen2.5-7B-Instruct", 
                 max_tokens = 4096,
                 stop = None,
                 temperature = 1,
                 top_p = 0.7,
                 top_k = 1,
                 frequency_penalty = 0,
                 n = 1,
                 key_idx = 0,
                #  max_prompt_length = 1024,
            ):

        self.payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "stream": False,
            "max_tokens": max_tokens,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "n": 1,
            "response_format": {"type": "text"},
            "tools": []
        }
        
        if model != "gpt-4o-mini":
            self.url = "https://api.siliconflow.cn/v1/chat/completions"
            keys = api_key_dict["silicon_flow_keys"]
            self.headers = {
                "Authorization": f"Bearer {keys[key_idx]}",
                "Content-Type": "application/json"
            }
            
        elif model == "gpt-4o-mini":
            if self.payload["temperature"] != 0:
                print("Warning: temperature is not 0, set to 0 if greedy decode")
                self.payload["temperature"] = 0
                ### OpenAI API donot have a top_k parameter, so we need to set temperature to 0
            self.HTTP_CLIENT = httpx.Client(proxy="http://127.0.0.1:8456")
            self.client = AzureOpenAI(
                api_key = api_key_dict['azure_key'],
                api_version = "2024-07-01-preview",
                azure_endpoint = api_key_dict['azure_endpoint'],
                http_client = self.HTTP_CLIENT,
                azure_deployment="gpt-4o-mini-2"
            )

        self.q_token = 0
        self.a_token = 0

    def reset_token(self):
        self.q_token = 0
        self.a_token = 0

    def get_token(self):
        return self.q_token, self.a_token

    def get_response(self):
        t0 = time.time()
        max_try = 10
        try_num = 0
        max_sleep = 20
        sleep_time = 1
        while True:
            try_num += 1
            if self.payload["model"] == "gpt-4o-mini":
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "user", "content": self.payload["messages"][0]["content"]}
                        ],
                        max_tokens=self.payload["max_tokens"],
                        stop=self.payload["stop"],
                        temperature=self.payload["temperature"],
                        top_p=self.payload["top_p"],
                        frequency_penalty=self.payload["frequency_penalty"],
                        n=self.payload["n"],
                        timeout=60,
                    )
                    content = response.choices[0].message.content
                    self.q_token += response.usage.prompt_tokens
                    self.a_token += response.usage.completion_tokens
                    if len(content) != 0:
                        response = {"choices": [{"message": {"content": content}}]}
                        break
                except Exception as e:
                    try:
                        if "content management policy" in str(e):
                            print("Warning: Blocked by content management policy")
                            response = {"choices": [{"message": {"content": ""}}]}
                            break
                    except:
                        pass
                    print("try_num: ", try_num, "error", e)
                    time.sleep(sleep_time)
                    sleep_time = min(sleep_time * 2, max_sleep)
                
                if try_num >= max_try:
                    print("Warning: TOO MANY TRIES, EMPTY RESPONSE")
                    response = {"choices": [{"message": {"content": ""}}]}
                    break
            
            else:
                try:
                    response = requests.request("POST", self.url, json=self.payload, headers=self.headers)
                    response = response.json()
                    content = response["choices"][0]["message"]["content"]
                    self.q_token += response["usage"]["prompt_tokens"]
                    self.a_token += response["usage"]["completion_tokens"]
                    if len(response) != 0:
                        break
                except Exception as e:
                    # print("try_num: ", try_num, "error", e)
                    time.sleep(sleep_time)
                    sleep_time = min(sleep_time * 2, max_sleep)

                if try_num >= max_try:
                    print("Warning: TOO MANY TRIES, EMPTY RESPONSE")
                    response = {"choices": [{"message": {"content": ""}}]}
                    break

        return response
    
    def get_text(self, content = ""):
        self.set_content(content)
        response = self.get_response()
        return response['choices'][0]['message']['content']
    
    def extract_state(self, res_string):
        ## a post process for extracting the score. there are 8 of them,
        state_key = ["A1", "A2", "A3", "B1", "B2", "B3", "B4", "C1"]
        state = {}
        for key in state_key:
            location = res_string.find(key)
            if location != -1 and location + 4 < len(res_string):
                char = res_string[location + 4]
                state[key] = char
                if char in ["1", "2", "3", "4", "5"]:
                    state[key] = int(char)
                else:
                    state[key] = -1
            else:
                state[key] = -1
        return state
    
    def set_content(self, content):
        self.payload["messages"][0]["content"] = content

    def print_usage(self):
        print(f"Prompt tokens: {self.q_token}")
        print(f"Completion tokens: {self.a_token}")
    

if __name__ == "__main__":
    llm = LLM_api(model="meta-llama/Meta-Llama-3.1-70B-Instruct", key_idx=6)
    text = llm.get_text(content="123, 321")
    print(text)