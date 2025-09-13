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

    def agent_generate(self, prompt, selected_sub_agent="Default", mode="chat", can_think=False, stream=False, include_header=True):
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

'''
Frontend handler for interactions with one or more agents.
'''

class Interaction:
    def __init__(self, mode:str,roster={"Default":Agent},do_stream=False):
        self.mode = mode
        self.roster = roster
        self.do_stream=do_stream
    #Experimental feature, may not work as intended
    def debate(self,turns:int, prompt:str, summarizer_agent=[Agent, str], summarizer_mode:int=0):
        i=0
        context=prompt
        temp_context=prompt
        while i<turns:
            for subagent,agent in self.roster.items():
                agent_output=agent.agent_generate(prompt=context,selected_sub_agent=subagent,mode=self.mode,stream=self.do_stream)
                temp_context+=subagent+":"+agent_output+'\n'
            if summarizer_mode==1:
                agent_output=self.summarizer_agent[0].agent_generate(prompt=temp_context,selected_sub_agent=self.summarizer_agent[1],mode=self.mode,stream=self.do_stream)
                context+=agent_output
            else:
                context+=temp_context
            temp_context=""
            i+=1
        return context
    #Experimental feature, may not work as intended
    def roundtable(self,turns:int, prompt:str, summarizer_agent=[Agent, str], summarizer_mode:int=0):
        i=0
        context=prompt
        while i<turns:
            for subagent,agent in self.roster.items():
                agent_output=agent.agent_generate(prompt=context,selected_sub_agent=subagent,mode=self.mode,stream=self.do_stream)
                context+=subagent+":"+agent_output+'\n'
            if summarizer_mode==1:
                agent_output=self.summarizer_agent[0].agent_generate(prompt=context,selected_sub_agent=self.summarizer_agent[1],mode=self.mode,stream=self.do_stream)
                context+=prompt+agent_output
            i+=1
        return context
    
    '''
    Single agent chat function. Allows user to interact with one sub-agent at a time.
    Has commands for selecting sub-agent, saving chat, clearing chat, and quitting.
    Saving chat is experimental and may not work as intended.
    '''
    def single_agent_chat(self, subagent:str="Default", username="User", machinename="AI", summarizer_agent=[Agent, str], summarizer_mode:int=0):
        try:
            agent = self.roster[subagent]
            context=f"{username}: "
            while True:
                try:
                    user_cmd=input("\nModel Prompt>>>")
                    #Activate commands if input starts with /
                    # e.g. Model Prompt>>>/ SELECT SUBAGENT <subagent name>
                    if user_cmd.split()[0]=="/":
                        match user_cmd.split()[1].upper():
                            case "SELECT":
                                match user_cmd.split()[2].upper():
                                    case "SUBAGENT":
                                        print(f"Changed subagent {subagent} running on {agent.model_path} to {user_cmd.split()[3].lower()} running on {self.roster[user_cmd.split()[3].lower()]}!")
                                        subagent=user_cmd.split()[3].lower()
                                    case "SEED":
                                        agent.seed=int(user_cmd.split()[3])
                                    case "SUMMARIZER":
                                        match user_cmd.split()[3].upper():
                                            case "0":
                                                summarizer_mode=0
                                                print(f"Summarizer turned ON using {summarizer_agent[0]} with sub-agent {summarizer_agent[1]}!")
                                            case "1":
                                                summarizer_mode=1
                                                print("Summarizer turned OFF!")
                                            case _:
                                                print("Command not recognized.")
                                    case _:
                                        print("Command not recognized.")
                            case "SHOW":
                                match user_cmd.split()[2].upper():
                                    case "CONTEXT":
                                        print(context)
                                    case _:
                                        print("Command not recognized.")
                            case "CLEAR":
                                match user_cmd.split()[2].upper():
                                    case "CHAT":
                                        context=f"{username}: "
                            case "QUIT":
                                exit()
                            case _:
                                print("Command not recognized.")
                    else:
                        context+=user_cmd
                        context+=f"\n{machinename}:"
                        if summarizer_mode==1:
                            response_buffer=agent.agent_generate(mode=self.mode,stream=self.do_stream,prompt=context,selected_sub_agent=subagent)
                            context+=summarizer_agent[0].agent_generate(prompt=response_buffer,selected_sub_agent=summarizer_agent[1],mode=self.mode,stream=False)
                        else:
                            context+=agent.agent_generate(mode=self.mode,stream=self.do_stream,prompt=context,selected_sub_agent=subagent)
                        context+=f"\n{username}: "
                except KeyboardInterrupt as k:
                    print(f"\nSTOPPED GENERATION FOR {subagent} BY USER INTERRUPT!")
                except Exception as e:
                    print(f"Error: {e} occured.")
        except ValueError as ve:
            print(f"Possibly more than one agent? Please try one then. ERROR CODE: {ve}")

    '''
    Allows user to interact with multiple agents at once.
    Each agent responds in turn to the user's input, and the context is built up over time.
    '''
    def multi_agent_chat(self, username="User", summarizer_agent=[Agent, str], summarizer_mode:int=0):
        try:
            context=f"{username}: "
            temp_contex=""
            while True:
                try:
                    user_cmd+=input("\nModel Prompt>>>")
                    #Activate commands if input starts with /
                    # e.g. Model Prompt>>>/ SELECT SUBAGENT <subagent name>
                    if user_cmd.split()[0]=="/":
                            match user_cmd.split()[1].upper():
                                case "SELECT":
                                    match user_cmd.split()[2].upper():
                                        case "SEED":
                                            agent.seed=int(user_cmd.split()[3])
                                        case "SUMMARIZER":
                                            match user_cmd.split()[3].upper():
                                                case "0":
                                                    summarizer_mode=0
                                                    print(f"Summarizer turned ON using {summarizer_agent[0]} with sub-agent {summarizer_agent[1]}!")
                                                case "1":
                                                    summarizer_mode=1
                                                    print("Summarizer turned OFF!")
                                                case _:
                                                    print("Command not recognized.")
                                        case _:
                                            print("Command not recognized.")
                                case "SHOW":
                                    match user_cmd.split()[2].upper():
                                        case "CONTEXT":
                                            print(context)
                                        case _:
                                            print("Command not recognized.")
                                case "CLEAR":
                                    match user_cmd.split()[2].upper():
                                        case "CHAT":
                                            context=f"{username}: "
                                        case _:
                                            print("Command not recognized.")
                                case "QUIT":
                                    exit()
                                case _:
                                    print("Command not recognized.")
                    else:
                        temp_context=context
                        for subagent,agent in self.roster.items():
                            temp_context+=f"\n{subagent}: "
                            #print(f"{subagent}: ")
                            temp_context+=agent.agent_generate(mode=self.mode,stream=self.do_stream,prompt=context,selected_sub_agent=subagent)
                            temp_contex+=f"\n{username}: "
                        context+=temp_contex
                except KeyboardInterrupt as k:
                    print(f"\nSTOPPED GENERATION BY USER INTERRUPT!")
                except Exception as e:
                    print(f"Error: {e} occured.")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    pass