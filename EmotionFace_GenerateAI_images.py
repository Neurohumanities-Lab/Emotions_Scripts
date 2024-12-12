import os
import numpy as np
import time
import openai
import torch
import csv
import tempfile
import cv2
import sys
import PIL
import argparse
import pandas as pd
import subprocess
import shutil
import socket
import asyncio
from datetime import datetime
from feat import Detector, utils
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from torch import autocast
from pythonosc import udp_client
from pythonosc import osc_server
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher


os.chdir(r'C:\PythonEscenas')

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default="emociones.csv", help='Nombre del archivo CSV')
parser.add_argument('--video', type=str, help='Nombre del archivo de video')
parser.add_argument('--gpt_prompts', type=str,default="false", help='Forma en que se generan los prompts')
parser.add_argument('--record', type=str, default="false", help='Grabar video: true o false')
parser.add_argument('--century', type=int, default=16, help='Definir siglo a adaptar')
args = parser.parse_args()

nombre_archivo = os.path.join('csv', args.csv)

if os.path.isfile(nombre_archivo):
    with open(nombre_archivo, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
        if data:
            last_row = data[-1]
            num_test = int(last_row[0]) + 1
        else:
            num_test = 1
else:
    num_test = 1


utils.set_torch_device(device='cuda')

IP_lap = socket.gethostbyname(socket.gethostname()) #IP as a server
PORT0 = 8001    #ReceivePort
dispatcher = Dispatcher()
df = pd.DataFrame(columns=['Timestamp','Detected emotion'])

emotions = ""
emocion_objetivo = ""
emociones = []
emocionFrec = ""
counter = 0
client1 = udp_client.SimpleUDPClient('148.220.164.180',11001) #LeftImage
client2 = udp_client.SimpleUDPClient('148.220.164.180',11002) #RightImage
client3 = udp_client.SimpleUDPClient('148.220.164.180',6999)  #FaceExpression
client4 = udp_client.SimpleUDPClient('148.220.164.180',5003)  #PureData

img_path_clock = os.path.join("imagenes","new_image_clock.jpg")
img_path_vase = os.path.join("imagenes","new_image_vase.jpg")
orig_path_clock = os.path.join("imagenes","originals","clock_black.jpg")
orig_path_vase =  os.path.join("imagenes","originals","vase1.jpg")

shutil.copy(orig_path_clock,img_path_clock)
shutil.copy(orig_path_vase,img_path_vase)

detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="svm",
        facepose_model="img2pose",
        device="cuda"
)

def crear_carpetas():
    if not os.path.exists('imagenes'):
        os.makedirs('imagenes')
    if not os.path.exists('csv'):
        os.makedirs('csv')

def init_camera():
    video_capture = cv2.VideoCapture(0)
    ret = video_capture.set(3,640)
    ret = video_capture.set(4,480)
    return video_capture

def acquire_image(video_capture, max_attempts=3):
    attempts = 0

    while attempts < max_attempts:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        
        if ret:
            scaled_rgb_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            scaled_rgb_frame = np.ascontiguousarray(scaled_rgb_frame[:, :, ::-1])
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, "temp_frame.jpg")
            cv2.imwrite(temp_file, scaled_rgb_frame)
            return frame, scaled_rgb_frame, temp_file
        else:
            attempts += 1

    print("--------No se pudo capturar la imagen / Fin del video------")
    return None, None, None

def show_frame(frame):
    # Display the resulting image frame in the PAC
    cv2.imshow('Video1',frame)

def find_face_emotion(frame):
    single_face_prediction = detector.detect_image(frame)
    data = single_face_prediction
    df = single_face_prediction.emotions
    if len(df) == 1 and df.isnull().all().all():
        emotion_list = []
    else:
        dict = df.idxmax(axis=1).to_dict()
        emotion_list = list(dict.values())
    return emotion_list, data

# def save_data(emociones,dataframe=None):
#     #Verificar si las listas están vacías
#     if not emociones:
#         return
#     #Obtener la fecha y hora actual
#     fecha_actual = datetime.now().strftime("%Y-%m-%d")
#     hora_actual = datetime.now().strftime("%H:%M:%S")

#     #Abrir el archivo CSV en modo de agregado ('a')
#     with open(nombre_archivo, 'a', newline='') as archivo_csv:
#         writer = csv.writer(archivo_csv)

#         #Escribir cada elemento en una nueva fila
#         for emocion in emociones:
#             if dataframe is not None:
#                 data_row = [num_test, fecha_actual, hora_actual, emocion]
#                 data_row += list(dataframe.iloc[0]) #Agregar datos del DataFrame
#                 writer.writerow(data_row)
#             else:
#                 writer.writerow([fecha_actual,hora_actual,emocion])
#     print("--------Datos guardados correctamente------------")

def load_image(file_path):
    return PIL.Image.open(file_path).convert("RGB")

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def infer(prompt, init_image, strength, img_path):
    #base_generated_folder = os.path.join("imagenes")
    #generated_image_name = f"Result_image.jpg"
    generated_image_path = img_path
    if init_image != None:
        init_image = init_image.resize((768, 768))
        init_image = preprocess(init_image)
        with autocast("cuda"):
            images = pipeimg(prompt=prompt, image=init_image, strength=strength, guidance_scale=7.5).images[0]#["sample"]
    else: 
        pass
    
    images.save(generated_image_path)
    #subprocess.Popen(["start", generated_image_path],shell=True)
    return images

def spanishEmo(emotionV):
    emotionEn = {
        "anger": "enojo",
        "fear": "miedo",
        "disgust": "asco",
        "happiness": "felicidad",
        "sadness": "tristeza",
        "surprise": "sorpresa",
        "neutral": "neutral"
    }
    return emotionEn[emotionV]

def generate_inpainting_prompt(emotion, element):
    emotion_values = {
        "anger": (-1, -1, -1),
        "fear": (-1, 1, -1),
        "disgust": (None, None, None),
        "happiness": (1, 1, 0),
        "sadness": (-1, -1, -1),
        "surprise": (0, 1, -1),
        "neutral": (0, 0, 0) 
    }
     # Create a dictionary with modifications for each element based on valence, arousal, and dominance values
    element_modifications = {
        "flower": {
            (-1, -1, -1): "Darker colors. The flower withers and shrinks. Petals and leaves harden. More pointed shapes and thorns.",
            (-1, 1, -1): "Darker colors. The flower blooms and grows. Petals and leaves harden. More pointed shapes and thorns.",
            (None, None, None): "Dull colors. The flower remains the same. Petals and leaves wrinkle. Irregular shapes.",
            (1, 1, 0): "Brighter colors. The flower blooms and grows. Petals and leaves soften. Rounded and smooth shapes.",
            (-1, -1, -1): "Darker colors. The flower withers and shrinks. Petals and leaves harden. More pointed shapes and thorns.",
            (0, 1, -1): "Varied colors. The flower blooms and grows. Petals and leaves harden. Unexpected and surprising shapes.",
            (0, 0, 0): "Neutral colors. The flower remains unchanged. No significant changes in shape or size."  # Prompt for "neutral" emotion
        },
        "hourglass": {
            (-1, -1, -1): "Darker colors. The hourglass becomes older, deteriorates. Time passes more slowly.",
            (-1, 1, -1): "Darker colors. The hourglass becomes older, deteriorates. Time passes more quickly.",
            (None, None, None): "Dull colors. The hourglass remains the same. Time passes randomly.",
            (1, 1, 0): "Brighter colors. The hourglass becomes more modern, new, and shiny. Time passes at the desired pace.",
            (-1, -1, -1): "Darker colors. The hourglass becomes older, deteriorates. Time passes more slowly.",
            (0, 1, -1): "Varied colors. The hourglass becomes randomly more modern or older. Time passes unpredictably.",
            (0, 0, 0): "Neutral colors. The hourglass remains unchanged. Time stands still."  # Prompt for "neutral" emotion
        }
    }
    #Get valence, arousal, and dominance values for the given emotion
    valence, arousal, dominance = emotion_values[emotion]

    #Get the modification for the given element based on valence, arousal, and dominance values
    modification = element_modifications[element][(valence,arousal,dominance)]

    #Create the prompt using the modification
    prompt = f"{modification}"

    return prompt
    
def generate_elements_dict(archivo_csv,mode):
    elements = {}
    columns = ['sadness', 'neutral', 'fear', 'happiness', 'surprise', 'anger', 'disgust']
    initial_df = pd.DataFrame(columns=columns, index=range(10))
    with open(archivo_csv , 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) == 0:
                break
            
            if len(row) >= 3:
                key = row[1]
                value = row[2]
                if (mode == 1):
                    elements[key] = {'value': value, 'count': 0,'dataframe': initial_df.copy()}
                elif(mode == 2):
                    elements[key] = {'value': value, 'count': 0,'prompt': ""}
    if not elements:
        return None 
    return elements

pipeimg = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

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

def save_data(emotionDetected):
    present = datetime.now()
    ts = datetime.timestamp(present)
    newrow = {'Timestamp': ts,
              'Detected emotion' : emotionDetected}
    df.loc[len(df)] = newrow
    data_to_append = [ts,emotionDetected] 

save = dispatcher.map("/Save", data_save)
emotion = dispatcher.map("/Selected Emotion", data_emotion)
subject = dispatcher.map("/Subject", data_subject)
scene = dispatcher.map("/Scene", data_scene)
date = dispatcher.map("/Date", data_date)
hour = dispatcher.map("/Hour", data_hour)

video_capture = init_camera()

img1 = cv2.imread(r"C:\PythonEscenas\imagenes\new_image_vase.jpg", cv2.IMREAD_ANYCOLOR) 
img2 = cv2.imread(r"C:\PythonEscenas\imagenes\new_image_clock.jpg", cv2.IMREAD_ANYCOLOR) 
img1 = cv2.resize(img1, (450,500), interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, (450,500), interpolation=cv2.INTER_AREA)  
x=1950     
cv2.imshow("RightImage",img1)
cv2.moveWindow('RightImage',x+450,0)
cv2.imshow("LeftImage",img2)
cv2.moveWindow('LeftImage',x,0)
cv2.resizeWindow('LeftImage',450,500)
cv2.resizeWindow('RightImage',450,500)

async def loop():
    counter=0
    while save != "True" and scene != "Scene_4":
        print(save,emotion,subject,scene,date,hour,counter)
        counter += 1
        await asyncio.sleep(0.5)

async def init_main():
    server = AsyncIOOSCUDPServer((IP_lap, PORT0), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving
    await loop()  # Enter main loop of program
    transport.close()  # Clean up serve endpoint

asyncio.run(init_main())

###################################################################
# START

#elements_dict = generate_elements_dict('descriptions.csv',2)
crear_carpetas()



cv2.waitKey(1000)
time.sleep(1)
client1.send_message("/LeftImgCounter",counter)
client2.send_message("/RightImgCounter",counter)
client3.send_message("/Face Expression"," ")

try:
    while (True):
        #############################################################
        # SENSING LAYER
        rgb_frame, scaled_rgb_frame, temp_file = acquire_image(video_capture)
        if rgb_frame is None:
            break
        #Emotion recognition
        face_emotions,data = find_face_emotion(temp_file)
        try:
            print(face_emotions[0])
            NoFace = False
            if len(emociones) < 7:
                emociones.append(face_emotions[0])
                print(emociones)
                save_data(face_emotions[0])
                emocionFrec = None
            else:
                emocionFrec = str(max(emociones, key=emociones.count))
                emociones=[]
        except IndexError:
            print('No face is detected.')
            NoFace = True

        if emocionFrec is not None and NoFace == False:
            emocionEnviada = spanishEmo(emocionFrec)
            print("La emoción más frecuente es: " + emocionEnviada)
            client3.send_message("/Face Expression",emocionEnviada)
            client4.send_message("/faceEmotion",emocionFrec)
            image_prompt1 = generate_inpainting_prompt(emocionFrec,'hourglass')
            image_prompt2 = generate_inpainting_prompt(emocionFrec,'flower')
            print(image_prompt1)
            print(image_prompt2)
            im_vase = Image.open(r"C:\PythonEscenas\imagenes\new_image_vase.jpg")
            im_clock = Image.open(r"C:\PythonEscenas\imagenes\new_image_clock.jpg")
            images = infer(image_prompt1, im_clock, 0.5, img_path_clock)
            images = infer(image_prompt2, im_vase, 0.5, img_path_vase)
            #try:
            #    cv2.destroyWindow("RightImage")
            #    cv2.destroyWindow("LeftImage")
            #except:
            #    pass
            img1 = cv2.imread(r"C:\PythonEscenas\imagenes\new_image_vase.jpg", cv2.IMREAD_ANYCOLOR) 
            img2 = cv2.imread(r"C:\PythonEscenas\imagenes\new_image_clock.jpg", cv2.IMREAD_ANYCOLOR) 
            #cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            #cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow('Video1',0,0) 
            #cv2.resizeWindow('Video',1300,800)
            #cv2.namedWindow('LeftImage', cv2.WINDOW_NORMAL)
            #cv2.setWindowProperty('LeftImage', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            img1 = cv2.resize(img1, (450,500), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (450,500), interpolation=cv2.INTER_AREA)  
            cv2.imshow("RightImage",img1)
            cv2.moveWindow('RightImage',x+450,0)
            cv2.imshow("LeftImage",img2)
            cv2.moveWindow('LeftImage',x,0)
            cv2.resizeWindow('LeftImage',450,500)
            cv2.resizeWindow('RightImage',450,500)

            # cv2.imshow("RightImage",img1)
            # cv2.moveWindow('RightImage',1850,0)
            # cv2.imshow("LeftImage",img2)
            
            cv2.waitKey(1500)
            counter += 1
            client1.send_message("/LeftImgCounter",counter) 
            client2.send_message("/RightImgCounter",counter)       
        else:
            pass     
        
        show_frame(rgb_frame)
        #############################################################

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # END OF THE GAME/LIFE
            break
        #lastPublication = time.time()
    filename = f'{subject}_{scene}_{emotion}_{date}_{hour}.csv'
    print(filename)
    df.to_csv(filename)
        
except KeyboardInterrupt:
    filename = f'{subject}_{scene}_{emotion}_{date}_{hour}.csv'
    print(filename)
    df.to_csv(filename)
    cv2.destroyAllWindows()
