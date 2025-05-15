# Generative AI
## Project Setup 
~~~
uv venv venv-genai
~~~
~~~
source venv-genai/bin/activate
~~~
~~~
uv pip install -r requirements.txt
~~~

## Install Binaries
~~~
brew install ffmpeg
~~~

## Add jupyter kernel (Run this command inside the venv)
#### Pre-requisite
~~~
uv pip install ipykernel
~~~
~~~
python -m ipykernel install --user --name=venv-genai --display-name "GenAI Env"
~~~
Restart the IDE once after running above commands

## Add Python debugger
~~~
Cmd + Shift + P 
~~~
~~~
Python: Select Interpreter
~~~
Select the venv or add the path in `Enter interpreter path`
~~~
python_venv_folder/bin/activate
~~~

## Multi Agent Frameworks
**AutoGen:** Direct agent-to-agent messaging with a built-in chat history and conversation management.

**Crew AI:** Task-based communication where outputs from one agent become inputs to another.

**LangGraph:** State-passing between nodes in a directed graph with explicit edge connections.
