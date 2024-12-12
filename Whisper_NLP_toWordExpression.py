import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
import re
import socket
import pandas as pd
import csv
import google.auth
import gspread
import pygsheets
import asyncio
from time import sleep
from pythonosc import osc_server
from pythonosc import udp_client
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from gensim.models import KeyedVectors
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from sys import platform
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.service_account import ServiceAccountCredentials

os.chdir(r'C:\PythonEscenas')

#Configuration
corpus_model = KeyedVectors.load(r"Models\fullcorpusES.bin")
IP = '148.220.164.180'
IP_lap = socket.gethostbyname(socket.gethostname()) #IP as a server
PORT0 = 8001    #ReceivePort
PORT1 = 7002    # 5 words to send
PORT2 = 8002    #Size of 5 worlds according to the correlation number
PORT3 = 9002    #list of the selected words by user

dispatcher = Dispatcher()
df = pd.DataFrame(columns=['Timestamp','Word','Correlation','NewWord'])
scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

credentials = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
clientgoogle = gspread.authorize(credentials)

spreadsheet = clientgoogle.open('CSV_scene02')
worksheet = spreadsheet.worksheet('CSV_scene02')  # Replace 'Sheet1' with your sheet's name

def data_save(address: str,*args):
    global save
    save = args[0]
    return save

def data_subject(address: str,*args):
    global subject
    subject = args[0]
    return subject

def data_scene(address: str,*args):
    global scene
    scene = args[0]
    return scene

def data_date(address: str,*args):
    global date
    date = args[0]
    return date

def data_hour(address: str,*args):
    global hour
    hour = args[0]
    return hour

def data_emotion(address: str,*args):
    global emotion
    emotion = args[0]
    return emotion

#Parser argument for whisper
parserWhisper = argparse.ArgumentParser()
parserWhisper.add_argument("--model", default="large", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
parserWhisper.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
parserWhisper.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
parserWhisper.add_argument("--record_timeout", default=1.5,
                        help="How real time the recording is in seconds.", type=float)
parserWhisper.add_argument("--phrase_timeout", default=0.1,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)  
argsWhisper = parserWhisper.parse_args()

#Parser argument for OSC with th 5 words
parserOSCwords = argparse.ArgumentParser()
parserOSCwords.add_argument("--ip", default=IP, help="The ip of the OSC server")
parserOSCwords.add_argument("--port", type=str, default=PORT1, help="The port the OSC server is listening ON")
argsOSCwords = parserOSCwords.parse_args()

#Parser argument for OSC with the size of the 5 words
parserOSCsize = argparse.ArgumentParser()
parserOSCsize.add_argument("--ip", default=IP, help="The ip of the OSC server")
parserOSCsize.add_argument("--port", type=str, default=PORT2, help="The port the OSC server is listening ON")
argsOSCsize = parserOSCsize.parse_args()

#Parser argument for OSC with selected words by user
parserOSCselectwords = argparse.ArgumentParser()
parserOSCselectwords.add_argument("--ip", default=IP, help="The ip of the OSC server")
parserOSCselectwords.add_argument("--port", type=str, default=PORT3, help="The port the OSC server is listening ON")
argsOSCselectwords = parserOSCselectwords.parse_args()

data_queue = Queue()
last_sample = bytes()
phrase_time = None
recorder = sr.Recognizer()
recorder.energy_threshold = argsWhisper.energy_threshold
recorder.dynamic_energy_threshold =False
source = sr.Microphone(sample_rate=16000)

#Load model
model = argsWhisper.model
if argsWhisper.model != "large" and not argsWhisper.non_english:
    model = model #+ ".en"
audio_model = whisper.load_model(model)
record_timeout = argsWhisper.record_timeout
phrase_timeout = argsWhisper.phrase_timeout
temp_file = NamedTemporaryFile().name
transcription = ['']
selectedlist = []

with source:
    recorder.adjust_for_ambient_noise(source)

def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
print("Model loaded.\n")

save = dispatcher.map("/Save", data_save)
emotion = dispatcher.map("/Selected Emotion", data_emotion)
subject = dispatcher.map("/Subject", data_subject)
scene = dispatcher.map("/Scene", data_scene)
date = dispatcher.map("/Date", data_date)
hour = dispatcher.map("/Hour", data_hour)

def osc_words(words,numbers):
    client = udp_client.SimpleUDPClient(argsOSCwords.ip,argsOSCwords.port)
    for i,w in enumerate(words, start=1):
        print("/word"+str(i),w)
        client.send_message("/word"+str(i),w)
    #client.send_message("/word2",words[1])
    #client.send_message("/word3",words[2])
    #client.send_message("/word4",words[3])
    #client.send_message("/word5",words[4])

    client = udp_client.SimpleUDPClient(argsOSCsize.ip, argsOSCsize.port)
    for i,w in enumerate(numbers):
        client.send_message("/size"+str(i), w)
    # client.send_message("/size0", numbers[0])
    # client.send_message("/size1", numbers[1])
    # client.send_message("/size2", numbers[2])
    # client.send_message("/size3", numbers[3])
    # client.send_message("/size4", numbers[4])

def osc_selected(word):
    client = udp_client.SimpleUDPClient(argsOSCselectwords.ip, argsOSCselectwords.port)
    client.send_message("/word1", word)

def test_w2v(emotion,n):
    math_result = corpus_model.most_similar(positive=emotion,topn=n)
    w_vals = [v[0] for v in math_result]
    n_vals = [v[1] for v in math_result]
    return w_vals, n_vals

def repeated(words,lista):
    for i in range(2):
        try: 
            indx = lista.index(words[i])
            print(indx)
            lista[indx] = newwords[2]
        except ValueError:
            print("Ok")

def limpiar_acentos(text):
	acentos = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n','ü':'u'} #'Á': 'A', 'E': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'}
	for acen in acentos:
		if acen in text:
			text = text.replace(acen, acentos[acen])
	return text

def save_data(subject,emotion,word,correlation,newWord):
    present = datetime.now()
    ts = datetime.timestamp(present)
    newrow = {'Timestamp': ts,
              'Word' : word,
              'Correlation': correlation,
              'NewWord': newWord}
    df.loc[len(df)] = newrow
    data_to_append = [ts,subject,emotion, word, correlation, newWord]  
    worksheet.append_row(data_to_append)

def reduce(numbers,n):
    for i in range(n):
        numbers[i] = numbers[i]*0.85
    return numbers

def minimum(numbers):
    minim = min(numbers)
    index = numbers.index(minim)
    return index

async def loop():
    counter=0
    while save != True and scene != "Scene_2":
        print(save,emotion,subject,scene,date,hour,counter)
        counter += 1
        await asyncio.sleep(0.5)

async def init_main():
    server = AsyncIOOSCUDPServer((IP_lap, PORT0), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving
    await loop()  # Enter main loop of program
    transport.close()  # Clean up serve endpoint

asyncio.run(init_main())

#emotion = input('Enter the emotion: ')
print(emotion + ' is the selected emotion.')

[words, numbers] = test_w2v(emotion,5)

print(words)
print(numbers)

osc_words(words,numbers)

while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available(), language="es",temperature=0.2, word_timestamps=True)
                text = result['text'].strip()
                
                text = re.sub(r'[^\w\s]','',text)
                text=text.lower()
                text = limpiar_acentos(text)
                
                if text != '':
                    try:
                        id=words.index(text)
                        try:
                            [newwords,newnumbers]=test_w2v(text,3)
                            repeated(newwords,words)
                            save_data(subject,emotion,words[id],numbers[id],'False')
                            numbers = reduce(numbers,5)
                            words[id] = newwords[0]
                            numbers[id] = newnumbers[0]
                            index = minimum(numbers)
                            words[index] = newwords[1]
                            numbers[index] = newnumbers[1]
                            osc_words(words,numbers)
                            osc_selected(text)
                            selectedlist.append(text)                         
                        except KeyError:
                            print(text + ' no esta presente en el vocabulario, intenta de nuevo')
                    except ValueError:
                        print(text + ' no esta en la lista.')
                        try:
                            [newwords,newnumbers]=test_w2v(text,3)
                            repeated(newwords,words)
                            save_data(subject,emotion,text,0,'True')
                            numbers = reduce(numbers,5)
                            index = minimum(numbers)
                            words[index] = newwords[1]
                            numbers[index] = newnumbers[1]
                            index = minimum(numbers)
                            words[index] = newwords[0]
                            numbers[index] = newnumbers[0]
                            osc_selected(text)
                            osc_words(words,numbers)
                            selectedlist.append(text)
                        #repeated(newwords,words)
                        except KeyError:
                            print(text + ' no esta presente en el vocabulario, intenta de nuevo')
                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                #os.system('cls' if os.name=='nt' else 'clear')
                #for line in transcription:
                #    print(line)
                # Flush stdout.
                #print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            filename = f'{subject}_{scene}_{emotion}_{date}_{hour}.csv'
            df.to_csv(filename)
            break

print("\n\nTranscription:")
for line in transcription:
    print(line)

print("\n\nSelected words:")
print(selectedlist)
