'''

Simple Agent handler that allows easy use and setup of llama-cpp and transformers LLMs for chat.
Made by Rohan Bhargava

'''

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed, TextIteratorStreamer
from datetime import datetime
import threading
import torch
from llama_cpp import Llama

class Agent:

    DEFAULT_HYPER_PARAMS={
        "max_new_tokens":128, 
        "temperature":0.7, 
        "top_p":1.0, 
        "top_k":25, 
        "do_sample":True, 
        "repetition_penalty":1.0
    }

    def __init__(self, model_path, device="cuda", seed=9, category="transformers"):
        self.model_path = model_path
        self.device = device
        self.seed = seed
        self.category = category
        self.sub_agents={
                    "Default":["You are a helpful AI assitant", 
                   Agent.DEFAULT_HYPER_PARAMS]}
        self.__generation_args__= {}
        self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    trust_remote_code=False,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path
        )
        set_seed(seed)

    '''
    Sets up sub-agents with their system prompts and hyperparameters.
    This allows one model to behave like multiple agents with different personalities and settings, saving on VRAM mainly.
    sub_agents should be a dictionary of the form:
    {
        "Sub-Agent Name":["System Prompt", {Hyperparameters as dictionary}]
    }
    '''
    def set_up_sub_agents(self, sub_agents={
        "Default":["You are a helpful AI assitant", 
                   DEFAULT_HYPER_PARAMS]}):
        self.sub_agents = sub_agents

    '''
    Backend function to generate output from the model with a selected sub-agent.
    There are three "modes" of operation:
    "chat" - Uses the chat template for models that support it
    "pipe" - Uses the transformers pipeline for text-generation
    "other" - Uses the simple generate function
    Can also stream output if desired.
    '''

    def agent_generate(self, prompt, selected_sub_agent="Default", mode="chat", can_think=False, stream=False, include_header=True, use_context_mgr_instead=False, context=[]):
        try:

            output=""
            if mode=="chat":
                message = [
                    {"role": "system", "content": f"{self.sub_agents[selected_sub_agent][0]}"},
                    #{"role": "system", "content":self.sub_agents[selected_sub_agent][0]},
                    {"role": "user", "content":prompt}
                    ]

                inputs = self.tokenizer.apply_chat_template(message, return_tensors="pt", thinking=can_think, return_dict=True, add_generation_prompt=True).to(self.device)

                if stream:
                    streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                    print(prompt, end="", flush=True)

                    thread =  threading.Thread(target=self.model.generate, kwargs=dict(**inputs, **self.sub_agents[selected_sub_agent][1], streamer=streamer))
                    thread.start()

                    for token_text in streamer:
                        output += token_text
                        print(token_text, end="", flush=True)

                    thread.join()
                else:
                    model_response = self.model.generate(**inputs, **self.sub_agents[selected_sub_agent][1])
                    #output = self.tokenizer.batch_decode(model_response)[0]
                    output = self.tokenizer.decode(model_response[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            elif mode == "pipe":
                message = [
                {"role": "system", "content":self.sub_agents[selected_sub_agent][0]},
                {"role": "user", "content":prompt}
                ]
        
                pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer
                )
                if stream:
                    streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True)
                    thread = threading.Thread(target=pipe, kwargs=dict(message, **self.sub_agents[selected_sub_agent][1], streamer=streamer))
                    thread.start()

                    for token_text in streamer:
                        output += token_text
                        print(token_text, end="", flush=True)

                    thread.join()
                else:
                    model_response=pipe(message, **self.sub_agents[selected_sub_agent][1])
                    output = model_response[0]['generated_text'][2]["content"]
            else:
                torch.set_default_device(self.device)
                message = self.sub_agents[selected_sub_agent][0]+prompt
                inputs = self.tokenizer(message, return_tensors="pt", return_attention_mask=False)

                if stream:
                    streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                    print(prompt, end="", flush=True)

                    thread =  threading.Thread(target=self.model.generate, kwargs=dict(**inputs, **self.sub_agents[selected_sub_agent][1], streamer=streamer))
                    thread.start()

                    for token_text in streamer:
                        output += token_text
                        print(token_text, end="", flush=True)

                    thread.join()
                else:
                    model_response = self.model.generate(**inputs, **self.sub_agents[selected_sub_agent][1])
                    output = self.tokenizer.batch_decode(model_response)[0]

            return str(output)
        except Exception as e:
            print(e)
            return e
        
class llamacppAgent:

    DEFAULT_HYPER_PARAMS={
        "max_new_tokens":128, 
        "temperature":0.7, 
        "top_p":1.0, 
        "top_k":25, 
        "do_sample":True, 
        "repetition_penalty":1.0
    }

    #Important to add all sub_agents that one may want to use to the roster paramter.
    def __init__(self, modelpath=None, repoid=None, file_name=None, context_length=4096, seed=9, layers_on_gpu=0):
        self.sub_agents={
                    "Default":["You are a helpful AI assitant", 
                   Agent.DEFAULT_HYPER_PARAMS]}
        self.__generation_args__= {}
        if modelpath==None:
            self.llm = Llama.from_pretrained(
                repo_id=repoid,
                filename=file_name,
                verbose=False,
                n_ctx=context_length,
                seed=seed,
                n_gpu_layers=layers_on_gpu
            )
        else:
            self.llm = Llama(
                model_path=modelpath,
                verbose=False,
                n_ctx=context_length,
                seed=seed,
                n_gpu_layers=layers_on_gpu
            )

    def set_up_sub_agents(self, sub_agents={
        "Default":["You are a helpful AI assitant", 
                   DEFAULT_HYPER_PARAMS]}):
        self.sub_agents = sub_agents

    def agent_generate(self, prompt, selected_sub_agent="Default", mode="chat", can_think=False, stream=False, save_output=False, include_header=True, save_location=None):
        try:
            chunks=self.llm.create_chat_completion(
                stream=stream,
                messages = [
                    {"role": "system", "content": self.sub_agents[selected_sub_agent][0]},
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.sub_agents[selected_sub_agent][1]["max_new_tokens"],
                temperature=self.sub_agents[selected_sub_agent][1]["temperature"],
                top_p=self.sub_agents[selected_sub_agent][1]["top_p"],
                top_k=self.sub_agents[selected_sub_agent][1]["top_k"],
                repeat_penalty=self.sub_agents[selected_sub_agent][1]["repetition_penalty"]
            )
            if stream:
                full_reply = ""
                for chunk in chunks:                       
                    delta = chunk["choices"][0]["delta"]   
                    if "content" not in delta:          
                        continue

                    token = delta["content"]
                    print(token, end="", flush=True)
                    full_reply += token
            else:
                full_reply=chunks["choices"][0]["message"]["content"]
            return str(full_reply)
        except Exception as e:
            return e

if __name__ == "__main__":
    pass