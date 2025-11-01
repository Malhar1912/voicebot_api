from fastapi import FastAPI, Request, Form
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
import pandas as pd
import os
import requests
from pydub import AudioSegment

app = FastAPI()

# --- Configure Gemini from environment variable ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Load all CSVs into one DataFrame ---
csv_dir = os.path.join(os.getcwd(), "data")
dataframes = []
for file in os.listdir(csv_dir):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(csv_dir, file))
        dataframes.append(df)
combined_df = pd.concat(dataframes, ignore_index=True)

# --- Helper: Search CSVs for relevant info ---
def search_property_info(query: str) -> str:
    try:
        results = combined_df[combined_df.apply(lambda row: row.astype(str).str.contains(query, case=False, na=False).any(), axis=1)]
        if not results.empty:
            sample = results.head(3).to_string(index=False)
            return f"I found the following information related to '{query}':\n{sample}"
        else:
            return f"Sorry, I couldn't find any information related to '{query}'."
    except Exception as e:
        return f"Error searching data: {str(e)}"

@app.post("/voice")
async def answer_call(request: Request):
    response = VoiceResponse()
    response.say("Hello! I am your property assistant. Please ask your question after the beep.")
    response.record(max_length=10, play_beep=True, action="/process_audio")
    return Response(content=str(response), media_type="application/xml")

@app.post("/process_audio")
async def process_audio(request: Request):
    form = await request.form()
    recording_url = form["RecordingUrl"]

    # Download the recorded audio from Twilio
    audio_data = requests.get(recording_url + ".wav").content
    with open("input.wav", "wb") as f:
        f.write(audio_data)

    # Convert to a consistent WAV format using ffmpeg
    sound = AudioSegment.from_wav("input.wav")
    sound.export("processed.wav", format="wav")

    # Speech Recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile("processed.wav") as source:
        audio = recognizer.record(source)
    try:
        user_text = recognizer.recognize_google(audio)
    except Exception:
        user_text = "Sorry, I could not understand your speech."

    print("👂 User said:", user_text)

    # Search CSV for info
    property_info = search_property_info(user_text)

    # Gemini Response
    prompt = f"The user asked: '{user_text}'. Using this info: {property_info}. Respond naturally and helpfully."
    reply = model.generate_content(prompt)
    response_text = reply.text if reply and hasattr(reply, "text") else "I'm sorry, I couldn't generate a response."

    print("🤖 AI reply:", response_text)

    # Convert text to speech
    tts = gTTS(response_text)
    tts.save("response.mp3")

    # Build TwiML response
    twiml = f"""
    <Response>
        <Play>https://your-domain.com/response.mp3</Play>
    </Response>
    """
    return Response(content=twiml, media_type="application/xml")
