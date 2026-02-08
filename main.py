from fastapi import Depends, FastAPI, HTTPException, status
from fastapi import UploadFile, File, Form
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv 
import os
import json

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

from ollama import Client
from huggingface_hub import InferenceClient

import tempfile

load_dotenv()

API_KEY = os.getenv("API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
CREDENTIALS_PATH = os.getenv("CREDENTIALS_PATH")

app = FastAPI()
header_scheme = APIKeyHeader(name="x-key")


def verify_api_key(api_key: str = Depends(header_scheme)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return api_key


@app.post("/query/")
async def query(audio: UploadFile = File(...), messages: str = Form(...), key: str = Depends(verify_api_key)):
    messages = json.loads(messages)

    audio_recognizer = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
    
    audio_bytes = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        output = audio_recognizer.automatic_speech_recognition(
            tmp.name,  # ファイルパスを渡す
            model="openai/whisper-large-v3"
        )
        query = output.text
        messages.append({"role": "user", "content": query})
    

    server_params = StdioServerParameters(
        command="npx", 
        args=["-y", "@cocal/google-calendar-mcp"],
        env={
            "GOOGLE_OAUTH_CREDENTIALS": CREDENTIALS_PATH
        }
    )

    # サーバ接続のためのクライアントストリームを確立
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 初期化処理（initialize）
            await session.initialize()

            response = await session.list_tools()
            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": tool.inputSchema["type"],
                        "required": tool.inputSchema["required"] if "required" in tool.inputSchema else [],
                        "properties": tool.inputSchema["properties"]
                    }
                }
            } for tool in response.tools]
            available_functions = [tool["function"]["name"] for tool in available_tools]
            
            client = Client(host='https://ollama.com', headers={'Authorization': 'Bearer ' + OLLAMA_API_KEY})
            for _ in range(10):
                response = client.chat('gpt-oss:20b-cloud', messages=messages, tools=available_tools)
                messages.append(response.message)
                if response.message.tool_calls:
                    for tc in response.message.tool_calls:
                        if tc.function.name in available_functions:
                            tool_name = tc.function.name
                            tool_args = tc.function.arguments
                            print(f"Calling {tc.function.name} with arguments {tc.function.arguments}")
                            result = await session.call_tool(tool_name, tool_args)
                            print(f"Result: {result}")
                            # add the tool result to the messages
                            for content in result.content:
                                messages.append({'role': 'tool', 'tool_name': tc.function.name, 'content': content.text})
                else:
                    # end the loop when there are no more tool calls
                    return [msg for msg in messages if msg["role"] != "tool"]
            return {"error": "tool loop exceeded"}