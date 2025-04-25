from pathlib import Path

from pydantic import BaseModel

import langchain
from fastapi import FastAPI, File, UploadFile
import uvicorn
import whisperx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
langchain.verbose = False


app = FastAPI()
client = ChatOpenAI(model="gpt-4")


def save_audio(file: UploadFile) -> str:
    upload_path = Path("uploads")
    upload_path.mkdir(exist_ok=True)
    file_location = upload_path / file.filename
    with open(file_location, "wb") as buffer:
        buffer.write(file.file.read())
    return str(file_location)


def transcribe_audio(path: str) -> str:
    model = whisperx.load_model("medium", device="cpu")
    audio = whisperx.load_audio(path)
    _result = model.transcribe(audio, batch_size=8)
    return _result["text"]


def analyze_emotion(_text: str) -> str:
    prompt = f"""
    Analyze the emotion expressed in the following journal entry. Respond with a single word or short phrase (e.g., happy, anxious, content, frustrated).

    Journal Entry:
    {_text}
    """
    messages = [
        HumanMessage(content=prompt)
    ]
    response = client.invoke(messages)
    return response.content


def generate_journal_prompt(emotion: str) -> str:
    prompt = f"""
    The user expressed the emotion: {emotion}.
    Suggest a reflective journal prompt to help process this feeling.
    """
    messages = [
        HumanMessage(content=prompt)
    ]
    response = client.invoke(messages)
    return response.content


class FastMCP:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.tools = {}

    def tool(self, name):
        def decorator(func):
            func.tool_name = name
            self.tools[name] = func
            return func
        return decorator


mcp = FastMCP("http://localhost:8000")


# Tools
@mcp.tool("analyze_emotion")
def emotion_tool(context, transcript: str):
    return analyze_emotion(transcript)


@mcp.tool("generate_prompt")
def prompt_tool(context, emotion: str):
    return generate_journal_prompt(emotion)


class MemoryBuffer:
    def __init__(self):
        self.history = {}

    def get(self, session_id):
        return self.history.get(session_id, [])

    def add(self, session_id, entry):
        self.history.setdefault(session_id, []).append(entry)


class MCPAgent:
    def __init__(self, tools, _memory):
        self.tools = {tool.tool_name: tool for tool in tools}
        self.memory = _memory

    def run(self, session_id, user_input):
        emotion = self.tools["analyze_emotion"](None, user_input)
        prompt = self.tools["generate_prompt"](None, emotion)
        self.memory.add(session_id, {"input": user_input, "emotion": emotion, "prompt": prompt})
        return {"emotion": emotion, "prompt": prompt}


class TextRequest(BaseModel):
    session_id: str
    user_input: str


memory = MemoryBuffer()
agent = MCPAgent(tools=[emotion_tool, prompt_tool], _memory=memory)


def run_mcp_agent(session_id: str, user_input: str):
    return agent.run(session_id=session_id, user_input=user_input)


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    path = save_audio(file)
    transcript = transcribe_audio(path)
    response = run_mcp_agent(session_id="user1", user_input=transcript)
    return {"transcript": transcript, "analysis": response}


@app.post("/mcp-analyze")
async def analyze_with_text(payload: TextRequest):
    _result = run_mcp_agent(session_id=payload.session_id, user_input=payload.user_input)
    return _result


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)