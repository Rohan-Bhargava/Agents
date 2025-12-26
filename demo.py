import Agents
import Interaction
llm=Agents.Agent("Model of your choice")
chat=Interaction.Interaction("chat",{"Default":llm},do_stream=True)
chat.single_agent_chat()