import Agents
llm=Agents.Agent("Model of your choice")
chat=Agents.Interaction("chat",{"Default":llm},do_stream=True)
chat.single_agent_chat()