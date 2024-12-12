import cv2
import mediapipe as mp
import time
import math
import openai
import numpy as np
import tempfile
import os
import shutil
import torch
import subprocess
import sys
import PIL
import matplotlib.pyplot as plt
from time import sleep
from pythonosc import udp_client
from datetime import datetime
from feat import Detector, utils
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from torch import autocast

os.chdir(r'C:\Users\Neurohumanities\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Neurohumanities\TOUCH-DESIGNER\4.2 (Apr 2024) (For Galleries) Transformations Final Delivery Backup')

#OSC IP and PORTS
IP = "10.12.181.191"
PORT1 = 6999        #Face expression
PORT2 = 1080        #Start percentage
PORT3 = 2000        #StartFlag
PORT4 = 1091        #Final stats
PORT5 = 11001       #Left Image 
PORT6 = 11002       #Right Image
PORT7 = 5003        #Emotion to Pure Data

clientFace = udp_client.SimpleUDPClient(IP, PORT1)
clientPercentage = udp_client.SimpleUDPClient(IP, PORT2)
clientStarFlag = udp_client.SimpleUDPClient(IP, PORT3)
clientStats = udp_client.SimpleUDPClient(IP, PORT4)
clientLeftImg = udp_client.SimpleUDPClient(IP, PORT5)
clientRightImg = udp_client.SimpleUDPClient(IP, PORT6)
clientPureData = udp_client.SimpleUDPClient(IP, PORT7)

counter = 0     #Counter increment when new image is generated

# MediaPipe Face model
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Stable difussion
pipeimg = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

utils.set_torch_device(device='cuda')

# Detector PyFeat
detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="svm",
        facepose_model="img2pose",
        device="cpu"
)

# Ellipse zone for face detection
axesLength = (120,80)  #containing major and minor axis of ellipse (major axis length, minor axis length).
center = (320,240)  #center coordinates of ellipse. The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
tolerancia = 30     #tolerancia para zona de deteccion

#Paths for images
img_path_clock = os.path.join("imagenes","new_image_clock.jpg")
img_path_vase = os.path.join("imagenes","new_image_vase.jpg")
orig_path_clock = os.path.join("imagenes","originals","clock_black.jpg")
orig_path_vase =  os.path.join("imagenes","originals","vase1.jpg")
#Copy original images to test folder
#shutil.copy(orig_path_clock,img_path_clock)
#shutil.copy(orig_path_vase,img_path_vase)

# FUNCTIONS
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

#Start the webcam, copy originals images to test folder and open the image's windows

video_capture = cv2.VideoCapture(1) # 0 for internal cam, 1 for external cam

#General loop
while (True):
    # Reset the images to the originals
    shutil.copy(orig_path_clock,img_path_clock)
    shutil.copy(orig_path_vase,img_path_vase)
    img1 = cv2.imread(r"C:\Users\Neurohumanities\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Neurohumanities\TOUCH-DESIGNER\4.2 (Apr 2024) (For Galleries) Transformations Final Delivery Backup\imagenes\new_image_vase.jpg", cv2.IMREAD_ANYCOLOR) 
    img2 = cv2.imread(r"C:\Users\Neurohumanities\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Neurohumanities\TOUCH-DESIGNER\4.2 (Apr 2024) (For Galleries) Transformations Final Delivery Backup\imagenes\new_image_clock.jpg", cv2.IMREAD_ANYCOLOR) 
    img1 = cv2.resize(img1, (450,500), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (450,500), interpolation=cv2.INTER_AREA) 
    cv2.imshow("RightImage",img1)
    cv2.imshow("LeftImage",img2)
    cv2.resizeWindow('LeftImage',450,500)
    cv2.resizeWindow('RightImage',450,500)
    cv2.moveWindow('LeftImage',960,0)
    cv2.moveWindow('RightImage',1410,0)

    # For webcam input:
    emociones = []
    total_emotions = []
    trigger_time = 3
    time_hand = trigger_time  #Time for the hand in the zone detection
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # First loop for trigger with hand
    #while (True):
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

            while video_capture.isOpened():
                init = time.time()
                success, image = video_capture.read()
                width = video_capture.get(3)
                height = video_capture.get(4)

                if not success:
                    print("Cannot open the camera.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(image, face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1))

                    x1 = face_landmarks.landmark[19].x*width
                    y1 = face_landmarks.landmark[19].y*height
                    
                    if (((x1>320-tolerancia) & (x1<320+tolerancia)) & ((y1>240-tolerancia) & (y1<240+tolerancia))):
                        time_hand = time_hand - (time.time() - init)
                        percent = (1 - time_hand/trigger_time)*100
                        print(time_hand)
                        cv2.putText(image,str(round(percent,0)),(50,50),font,1,(0,255,255),2,cv2.LINE_4)
                        clientPercentage.send_message("/StartPercentage", round(percent))
                    else:
                        print('Cara fuera de la zona')
                        time_hand = trigger_time
                        clientPercentage.send_message("/StartPercentage", 0)

                    if time_hand <= 0:
                        time_hand = 0
                        clientPercentage.send_message("/StartPercentage", 0)
                        clientStarFlag.send_message("/StartFlag", 1)
                        sleep(0.3)
                        clientStarFlag.send_message("/StartFlag", 0)
                        break
                
                cv2.ellipse(image,center,axesLength,90,0,360,(0,0,255),2)
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('Facemesh', cv2.flip(image, 1))
                cv2.moveWindow('Facemesh',980,540)
                cv2.waitKey(5)
                #if cv2.waitKey(5) & 0xFF == 27:
                #    break
    #video_capture.release()
    #cv2.destroyAllWindows()

    #Second loop (Vanitas sceneario)
    #Copy original images to test folder
    #shutil.copy(orig_path_clock,img_path_clock)
    #shutil.copy(orig_path_vase,img_path_vase)
    #img1 = cv2.imread(r"C:\Users\alex_\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\NEUROhumanities\Scene_04\imagenes\new_image_vase.jpg", cv2.IMREAD_ANYCOLOR) 
    #img2 = cv2.imread(r"C:\Users\alex_\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\NEUROhumanities\Scene_04\imagenes\new_image_clock.jpg", cv2.IMREAD_ANYCOLOR) 
    #img1 = cv2.resize(img1, (900,1000), interpolation=cv2.INTER_AREA)
    #img2 = cv2.resize(img2, (900,1000), interpolation=cv2.INTER_AREA)  
    #x=1950     
    #cv2.imshow("RightImage",img1)
    #cv2.moveWindow('RightImage',x+950,0)
    #cv2.imshow("LeftImage",img2)
    #cv2.moveWindow('LeftImage',x,0)
    #cv2.resizeWindow('LeftImage',900,1000)
    #cv2.resizeWindow('RightImage',900,1000)
    cv2.waitKey(1000)
    time.sleep(1)
    counter += 1
    clientLeftImg.send_message("/LeftImgCounter",counter)
    clientRightImg.send_message("/RightImgCounter",counter)
    while (True):
        init = time.time()
        # SENSING LAYER
        rgb_frame, scaled_rgb_frame, temp_file = acquire_image(video_capture)
        if rgb_frame is None:
            break
        #Emotion recognition
        face_emotions, data = find_face_emotion(temp_file)
        try:
            print(face_emotions[0])
            NoFace = False
            if len(emociones)<7:
                emociones.append(face_emotions[0])
                print(emociones)
                emocionFrec = None
                total_emotions.append(face_emotions[0])
            else:
                emocionFrec = str(max(emociones, key=emociones.count))
                emociones = []
        except IndexError:
            print('No face is detected.')
            NoFace = True

        if emocionFrec is not None and NoFace == False:
            emocionEnviada = spanishEmo(emocionFrec)
            print("La emoción más frecuente es: " + emocionEnviada)
            clientFace.send_message("/Face Expression",emocionEnviada)
            clientPureData.send_message("/faceEmotion",emocionFrec)
            image_prompt1 = generate_inpainting_prompt(emocionFrec,'hourglass')
            image_prompt2 = generate_inpainting_prompt(emocionFrec,'flower')
            im_vase = Image.open(r"C:\Users\Neurohumanities\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Neurohumanities\TOUCH-DESIGNER\4.2 (Apr 2024) (For Galleries) Transformations Final Delivery Backup\imagenes\new_image_vase.jpg")
            im_clock = Image.open(r"C:\Users\Neurohumanities\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Neurohumanities\TOUCH-DESIGNER\4.2 (Apr 2024) (For Galleries) Transformations Final Delivery Backup\imagenes\new_image_clock.jpg")
            images = infer(image_prompt1, im_clock, 0.5, img_path_clock)
            images = infer(image_prompt2, im_vase, 0.5, img_path_vase)
            #try:
            #    cv2.destroyWindow("RightImage")
            #    cv2.destroyWindow("LeftImage")
            #except:
            #    pass
            img1 = cv2.imread(r"C:\Users\Neurohumanities\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Neurohumanities\TOUCH-DESIGNER\4.2 (Apr 2024) (For Galleries) Transformations Final Delivery Backup\imagenes\new_image_vase.jpg", cv2.IMREAD_ANYCOLOR) 
            img2 = cv2.imread(r"C:\Users\Neurohumanities\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Neurohumanities\TOUCH-DESIGNER\4.2 (Apr 2024) (For Galleries) Transformations Final Delivery Backup\imagenes\new_image_clock.jpg", cv2.IMREAD_ANYCOLOR)
            img1 = cv2.resize(img1, (450,500), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (450,500), interpolation=cv2.INTER_AREA) 
            cv2.imshow("RightImage",img1)
            cv2.imshow("LeftImage",img2)
            #cv2.resizeWindow('LeftImage',450,500)
            #cv2.resizeWindow('RightImage',450,500)
            #cv2.moveWindow('LeftImage',960,0)
            #cv2.moveWindow('RightImage',1410,0)
            cv2.waitKey(100)
            counter += 1
            clientLeftImg.send_message("/LeftImgCounter",counter) 
            clientRightImg.send_message("/RightImgCounter",counter)  
        time_hand =  time_hand + (time.time() - init)
        print(time_hand)
        if time_hand >= 100:
            time_hand = 0
            break

    while (True):
        Total = len(total_emotions)
        Total_happines = round((total_emotions.count('happiness') / Total) * 100)
        Total_fear = round((total_emotions.count('fear') / Total) * 100)
        Total_anger = round((total_emotions.count('anger') / Total) * 100)
        Total_neutral = round((total_emotions.count('neutral') / Total) * 100)
        Total_sadness = round((total_emotions.count('sadness') / Total) * 100)
        Total_surprise = round((total_emotions.count('surprise') / Total) * 100)
        Total_disgust = round((total_emotions.count('disgust') / Total) * 100)

        clientStats.send_message("/Stat1", "Felicidad: " + str(Total_happines) + "%")
        clientStats.send_message("/Stat2", "Miedo: " + str(Total_fear) + "%")
        clientStats.send_message("/Stat3", "Enojo: " + str(Total_anger) + "%")
        clientStats.send_message("/Stat4", "Neutral: " + str(Total_neutral) + "%")
        clientStats.send_message("/Stat5", "Tristeza: " + str(Total_sadness) + "%")
        clientStats.send_message("/Stat6", "Sorpresa: " + str(Total_surprise) + "%")
        clientStats.send_message("/Stat7", "Asco: " + str(Total_disgust) + "%")
        clientStats.send_message("/DateandTime", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        cv2.waitKey(10000) 
        break 



