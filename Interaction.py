import Agents

'''
Frontend handler for interactions with one or more agents.
'''

class Interaction:
    def __init__(self, mode:str,roster={"Default":Agents.Agent},do_stream=False):
        self.mode = mode
        self.roster = roster
        self.do_stream=do_stream
    #Experimental feature, may not work as intended
    def debate(self,turns:int, prompt:str, summarizer_agent=[Agents.Agent, str], summarizer_mode:int=0):
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
    def roundtable(self,turns:int, prompt:str, summarizer_agent=[Agents.Agent, str], summarizer_mode:int=0):
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
    def single_agent_chat(self, subagent:str="Default", username="User", machinename="AI", summarizer_agent=[Agents.Agent, str], summarizer_mode:int=0):
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
                                        print(f"Changed subagent {subagent} to {user_cmd.split()[3].lower()}!")
                                        subagent=user_cmd.split()[3].lower()
                                    case "SEED":
                                        agent.seed=int(user_cmd.split()[3])
                                    case "SUMMARIZER":
                                        match user_cmd.split()[3].upper():
                                            case "0":
                                                summarizer_mode=0
                                                print(f"Summarizer turned OFF!")
                                            case "1":
                                                summarizer_mode=1
                                                print(f"Summarizer turned ON using {summarizer_agent[0]} with sub-agent {summarizer_agent[1]}!")
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
    def multi_agent_chat(self, username="User", summarizer_agent=[Agents.Agent, str], summarizer_mode:int=0, explicit_print_subagent_names:bool=False):
        try:
            while True:
                try:
                    user_cmd=input("\nModel Prompt>>>")
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
                                                    print(f"Summarizer turned OFF!")
                                                case "1":
                                                    summarizer_mode=1
                                                    print(f"Summarizer turned ON using {summarizer_agent[0]} with sub-agent {summarizer_agent[1]}!\n Summarizing mode set to summarize all agent responses at once!")
                                                case "2":
                                                    summarizer_mode=2
                                                    print(f"Summarizer turned ON using {summarizer_agent[0]} with sub-agent {summarizer_agent[1]}!\n Summarizing mode set to summarize each agent responses one at a time!")
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
                        context=f"{username}: "
                        context+=user_cmd
                        buffer_contex_responses_only=""
                        for subagent,agent in self.roster.items():
                            buffer_contex_responses_only+=f"\n{subagent}: "
                            if explicit_print_subagent_names:
                                print(f"{subagent}: ")
                            if summarizer_mode==2:
                                individual_response=agent.agent_generate(mode=self.mode,stream=self.do_stream,prompt=context,selected_sub_agent=subagent)
                                buffer_contex_responses_only+=summarizer_agent[0].agent_generate(mode=self.mode,stream=False,prompt=individual_response,selected_sub_agent=summarizer_agent[1])
                            else:
                                buffer_contex_responses_only+=agent.agent_generate(mode=self.mode,stream=self.do_stream,prompt=context,selected_sub_agent=subagent)
                        if summarizer_mode==1:
                            context+=summarizer_agent[0].agent_generate(mode=self.mode,stream=False,prompt=buffer_contex_responses_only,selected_sub_agent=summarizer_agent[1])
                        else:
                            context+=buffer_contex_responses_only
                except KeyboardInterrupt as k:
                    print(f"\nSTOPPED GENERATION BY USER INTERRUPT!")
                except Exception as e:
                    print(f"Error: {e} occured.")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    pass