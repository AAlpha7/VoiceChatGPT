"""
SUMMARY: Building an AI Chatbot with Google Cloud Speech-to-Text, OpenAI GPT-3, and Google Cloud Text-to-Speech for Voice Interaction

DESCRIPTION:

This code is an AI chatbot that uses Google Cloud Speech-to-Text, OpenAI's GPT-3, and Google Cloud Text-to-Speech to respond to a user's voice input.

The transcribe_speech() function records audio from the microphone and then transcribes the audio to text using Google Cloud Speech-to-Text. The resulting transcript is returned.

The OpenAI API key is loaded from a file using the openai_key_file_path variable and the conversation history is stored in the conversation string. The ask(newQuestion) function uses the conversation history and OpenAI's GPT-3 to generate a response to the user's input.

The text_to_mp3(text, voice_name, filename) function takes a text string, a voice name, and a filename and generates an MP3 audio file using Google Cloud Text-to-Speech. The play_mp3(filename) function plays an MP3 audio file using pygame.

The main loop of the code repeatedly records audio from the microphone, generates a response from the AI chatbot, and plays the response as audio. The conversation history is updated after each iteration and displayed after the chat ends. The chat ends when the user says "结束对话" (end conversation).

PLEASE NOTE:
1. The code is programed to work with Chinese. To use the code with a different language, you will need to change the language code in the transcribe_speech() function and the voice name in the text_to_mp3() function.
- Language codes: https://cloud.google.com/speech-to-text/docs/languages
- Voice names: https://cloud.google.com/text-to-speech/docs/voices
2. The command to end the conversasion is also Chinese ("结束对话"). To change the command, you will need to change the if statement in the main loop.
3. You will need to create a Google Cloud account and enable the Speech-to-Text, Text-to-Speech, and Translate APIs. You will also need to create an OpenAI account and get an API key.
- Google Cloud: https://cloud.google.com/
- OpenAI: https://openai.com/

ENJOY!
"""

import io
import os
import sys

from google.cloud import speech_v1
from google.cloud.speech_v1 import types
import pyaudio
import wave

import openai

# ---- Get Google Application Credentials and OpenAI API Key ----

if len(sys.argv) != 3:
    print("Usage: python VoiceChatGPT.py [google_creds_json_path] [openai_key_file_path]")
    sys.exit(1)

google_creds_json_path = sys.argv[1]
openai_key_file_path = sys.argv[2]

if not os.path.isfile(google_creds_json_path):
    print(f"Error: {google_creds_json_path} is not a file")
    sys.exit(1)

if not os.path.isfile(openai_key_file_path):
    print(f"Error: {openai_key_file_path} is not a file")
    sys.exit(1)

# Set Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_json_path

# Get OpenAI API Key
with open(openai_key_file_path, 'r') as file:
    openai.api_key = file.read().replace('\n', '')

# ---- Define a function to recognize speech from the microphone ----

import speech_recognition as sr
import os
from google.cloud import speech_v1
from google.cloud.speech_v1 import types

def transcribe_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        #print("请开始讲话...")
        audio = r.listen(source, timeout=30)
        #print("识别中，请稍等...")
        
    client = speech_v1.SpeechClient()

    config = types.RecognitionConfig(
        encoding=types.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='zh-CN'
    )

    audio = types.RecognitionAudio(content=audio.get_wav_data())

    response = client.recognize(config = config, audio = audio)

    # 遍历每一个识别结果，选择置信度最高的
    best_transcript = ''
    best_confidence = -1.0
    for result in response.results:
        transcript = result.alternatives[0].transcript
        confidence = result.alternatives[0].confidence
        #print('识别结果: {}'.format(transcript))
        #print('置信度: {}'.format(confidence))
        if confidence > best_confidence:
            best_transcript = transcript
            best_confidence = confidence
    
    # 如果没有检测到任何语音输入，则返回空字符串
    if not best_transcript and r.energy_threshold < 400:
        return ''
    
    return best_transcript



# ---- Define a function to get a response from OpenAI GPT-3 ----

def ask(newQuestion):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=conversation,
        temperature=0.5,
        max_tokens=1000,
        top_p=0.3,
        frequency_penalty=0.5,
        presence_penalty=0,
        stop=[" Human:", " AI:"]
    )

    answerText = response["choices"][0]["text"]
    return answerText

# ---- Define a function to convert text to speech ----

import pygame
import google.cloud.texttospeech as tts

def text_to_mp3(text: str, voice_name: str = "cmn-CN-Wavenet-A", filename: str = "answer.mp3"):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input, voice=voice_params, audio_config=audio_config
    )

    with open(filename, "wb") as out:
        out.write(response.audio_content)

def play_mp3(filename: str):
    pygame.mixer.init(frequency=22050)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play(0, 0.0, 1)
    while pygame.mixer.music.get_busy() == True:
        continue
    pygame.mixer.quit()

# ---- Main loop ----

conversation = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\n"
voiceText = ""
tmpResponseVoiceFileName = f"answer.mp3"
while "结束对话" not in voiceText:
    #voiceText = recognize_microphone()
    voiceText = transcribe_speech()
    if voiceText.strip() == "":
        continue
    elif "结束对话" in voiceText:
        break

    conversation = conversation + "Human: " + voiceText + "\nAI: "
    print("YOU: " + voiceText)
    answerText = ask(conversation)
    conversation = conversation + answerText + "\n"
    print("AI: " + answerText)

    text_to_mp3(answerText, filename=tmpResponseVoiceFileName)
    play_mp3(tmpResponseVoiceFileName)

print("= 对话记录 =")
print(conversation)