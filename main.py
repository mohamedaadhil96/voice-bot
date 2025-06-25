import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from langchain_openai import AzureChatOpenAI
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
PORT = int(os.getenv('PORT', 5050))

# Load Azure OpenAI model
def load_llm_model():
    return AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        temperature=0.0
    )

llm = load_llm_model()

# System settings
SYSTEM_MESSAGE = (
    "You are an AI assistant acting as a medical center receptionist. "
    "First greet the caller and ask if they are looking to schedule an appointment. "
    "Collect: Doctor's name, caller's name, phone number, and preferred appointment date/time. "
    "Once all info is gathered, respond with the following JSON:\n"
    '{\n  "docname": "",\n  "name": "",\n  "phone": "",\n  "appointment_datetime": ""\n}\n'
    "Only return valid JSON. Do not include explanations."
)

VOICE = "alloy"
LOG_EVENT_TYPES = ['error', 'response.done', 'input_audio_buffer.speech_started']
SHOW_TIMING_MATH = False

# Initialize FastAPI app
app = FastAPI()

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Application is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Respond to Twilio call with a stream instruction."""
    response = VoiceResponse()
    response.say("Please wait while we connect your call to the Medical Centre.", voice="alice")
    response.pause(length=1)
    response.say("Okay, you can start talking.", voice="alice")
    host = request.url.hostname or "your-domain.com"
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    print("WebSocket connected")
    await websocket.accept()

    async with websockets.connect(
        AZURE_OPENAI_ENDPOINT,
        extra_headers={"api-key": AZURE_OPENAI_API_KEY}
    ) as openai_ws:
        await initialize_session(openai_ws)

        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None

        async def receive_from_twilio():
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data["event"] == "media":
                        latest_media_timestamp = int(data["media"]["timestamp"])
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": data["media"]["payload"]
                        }))
                    elif data["event"] == "start":
                        stream_sid = data["start"]["streamSid"]
                        print(f"Stream started: {stream_sid}")
                        response_start_timestamp_twilio = 0
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data["event"] == "mark" and mark_queue:
                        mark_queue.pop(0)
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            nonlocal last_assistant_item, response_start_timestamp_twilio
            try:
                async for message in openai_ws:
                    response = json.loads(message)
                    if response["type"] in LOG_EVENT_TYPES:
                        print(f"[Azure Event] {response['type']}")

                    if response.get("type") == "response.audio.delta":
                        audio_payload = base64.b64encode(base64.b64decode(response["delta"])).decode()
                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": audio_payload}
                        })

                        if response_start_timestamp_twilio is None:
                            response_start_timestamp_twilio = latest_media_timestamp

                        if response.get("item_id"):
                            last_assistant_item = response["item_id"]

                        await send_mark(websocket, stream_sid)

                    if response.get("type") == "input_audio_buffer.speech_started":
                        print("Caller started speaking")
                        if last_assistant_item:
                            await handle_speech_started_event()
            except Exception as e:
                print("Error in send_to_twilio:", e)

        async def handle_speech_started_event():
            nonlocal last_assistant_item, response_start_timestamp_twilio
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed = latest_media_timestamp - response_start_timestamp_twilio
                if last_assistant_item:
                    await openai_ws.send(json.dumps({
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed
                    }))
                await websocket.send_json({"event": "clear", "streamSid": stream_sid})
                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                await connection.send_json({
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                })
                mark_queue.append("responsePart")

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

# Send first AI message
async def send_initial_conversation_item(openai_ws):
    await openai_ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello there! I am Jane from Medical Centre. How can I assist you today?"}]
        }
    }))
    await openai_ws.send(json.dumps({"type": "response.create"}))

# Configure session for Azure OpenAI
async def initialize_session(openai_ws):
    await openai_ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8
        }
    }))
    await send_initial_conversation_item(openai_ws)

# Run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
