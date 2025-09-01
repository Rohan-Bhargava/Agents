I created a simple agent handler for quickly deploying LLMs from the transformers library in both GGUF and .safetensors format.
This handler was created to speed up my own learning of LLMs.

An LLM is initlized as an Agent (or llamacppAgent) object. Each Agent object can have multiple "sub-agents" which are essentially personalities of one LLM.
From there, each Agent object then can be placed as an argument for an Interaction object.
Interaction objects allows the simple chat with a single sub-agent (chatting with a single agent).
I am expanding the Interaction class to include more autonomous uses of LLM Agents.
For example, I currently have multi_agent_chat() method which allows multiple sub-agents to each respond to single query.

Please see demo.py to see how to use this library.
This library is not a pip package yet so Agents.py will need to be in the same directory as the rest of the code.

This project will be periodically updated.
Any feedback, suggestions, or collaboration is welcome!