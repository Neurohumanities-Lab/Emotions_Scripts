# Pose Marks
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series

# OSC
import argparse
from random import randint
from time import sleep
from pythonosc import udp_client

# Math
import random

IP = "192.168.0.158"  
PORT1 = 5001
PORT2 = 5002
PORT3 = 5003
PORT4 = 5004

PORT5 = 5100


parser = argparse.ArgumentParser()
parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
parser.add_argument("--port", type=float, default=PORT1, help="The port the OSC server is listening on")
args = parser.parse_args()

client = udp_client.SimpleUDPClient(args.ip, args.port)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

BODY_PARTS = ['Nose', 'LInnerEye', 'LEye', 'LEyeOut', 'REyeInner', 'REye', 'REyeOut', 'LEar', 
              'REar', 'MouthL', 'MouthR', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist', 
              'LPinky', 'RPinky', 'LIndex', 'RIndex', 'LThumb', 'RThumb', 'LHip', 'RHip', 'LKnee', 'RKnee',
              'LAnkle', 'RAnkle', 'LHeel', 'RHeel', 'LFootIndex', 'RFootIndex']

  
def plot_coordinate(results, resolution):

  posefinal = np.array(
    [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)

  key_points = posefinal
  key_points_data = []

  for i in range(0, 33*4, 4): # 4 por x, y, z y visibility, 33 por los 33 puntos
      joint_coord = key_points[i], key_points[i + 1], key_points[i + 2], key_points[i + 3]
      if key_points[i + 3] > 0.5: #visibilidad mayor a 0.5
        key_points_data.append(joint_coord)
      else:
        key_points_data.append(0)

  #print(key_points)
  frame = Series(data=key_points_data, index=BODY_PARTS)
 
  coord_x_vector, coord_y_vector, coord_z_vector = [], [], []

  for marker in BODY_PARTS:
      tuple = str(frame[marker]).strip("()\n").replace(' ', '').split(',')
      if len(tuple) != 1:
          coord_x_vector.append((float(tuple[0])*resolution[0]-1)/resolution[0]) 
          coord_y_vector.append((resolution[1] - float(tuple[1])*resolution[1] -1)/resolution[1])
          coord_z_vector.append(float(tuple[2]))
      else:
        pass 
  #print(coord_x_vector)
        
  return True, coord_x_vector, coord_y_vector

cap = cv2.VideoCapture(0)
cam_resolution = (1920, 1080)

ph, = plt.plot(-10, -10, marker='o', color='red', linestyle='')
ax = plt.gca()
#ax.set_xlim([0, cam_resolution[0]])
ax.set_xlim([0, 1])
#ax.set_ylim([0, cam_resolution[1]])
ax.set_ylim([0, 1])
ax.set_xlabel('X coordinate (Pixels)')
ax.set_ylabel('Y coordinate (Pixels)')
ax.set_title('Real-time Mediapipe 2D key points with webcam')

frame_id = 1
refresh_rate = 0.01

#Starting Positions in TouchDesigner
#LX = -0.6115
#LY = -0.2595
#RX = 0.6115
#RY = -0.2595

headX = 0.395
headY = 0.395
leftWristX = 0.655
leftWristY = 0.395
rightWristX = 0.345
rightWristY = 0.395
waistX = 0.395
waistY = 0.395
leftAnkleX = 0.395
leftAnkleY = 0.395
rightAnkleX = 0.395
rightAnkleY = 0.395

print("-------------------------------------------")
print("Video input: Camera.")
print("Displaying Video Stream.")
print("Sending OSC messages on IP " + str(IP) + " and ports 5001-5004.")
print("Stream includes X and Y Coordinates for Head, Left/Right Wrists, Waist, and Left/Right Ankles.")
print("Coordinates for Points only sent if Body Part is present in input.")
print("Refresh Rate = " + str(refresh_rate) + ".")
print("-------------------------------------------")
print("Developed by Jesús Tamez-Duque.")
print("-------------------------------------------")

# Media Pipe Settings
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    c, coord_x, coord_y = plot_coordinate(results, cam_resolution)
    if c:
      
      ph.set_xdata(coord_x)
      ph.set_ydata(coord_y)


#Definition of Point Coordinates

      # 0 Define Head Coordinates

      Index = 0
      if len(coord_x) and len(coord_y) > Index:
        
        headX = round(float(coord_x[Index]),5)
        headY = round(float(coord_y[Index]),5)

      else:
        pass
      
      # 15 Define Left Wrist Coordinates

      Index = 15
      if len(coord_x) and len(coord_y) > Index:

        leftWristX = round(float(coord_x[Index]),5)
        leftWristY = round(float(coord_y[Index]),5)

      else:
        pass
    
      # 16 Define Rigth Wrist Coordinates

      Index = 16
      if len(coord_x) and len(coord_y) > Index:

        rightWristX = round(float(coord_x[Index]),5)
        rightWristY = round(float(coord_y[Index]),5)

      else:
        pass

      # 23 / 24 Define Waist Coordinates

      Index1 = 23
      Index2 = 24
      if len(coord_x) and len(coord_y) > Index2:

        leftWaistX = round(float(coord_x[Index1]),5)
        leftWaistY = round(float(coord_y[Index1]),5)
        rightWaistX = round(float(coord_x[Index2]),5)
        rightWaistY = round(float(coord_y[Index2]),5)

        waistX = (leftWaistX+rightWaistX) / 2
        waistY = (leftWaistY+rightWaistY) / 2
      
      else:
        pass

      # 27 Define Left Ankle Coordinates

      Index = 27
      if len(coord_x) and len(coord_y) > Index:

        leftAnkleX = round(float(coord_x[Index]),5)
        leftAnkleY = round(float(coord_y[Index]),5)

      else:
        pass
    
      # 28 Define Right Ankle Coordinates

      Index = 28
      if len(coord_x) and len(coord_y) > Index:

        rightAnkleX = round(float(coord_x[Index]),5)
        rightAnkleY = round(float(coord_y[Index]),5)

      else:
        pass


#Definition of Activation Values

      headActivationO = round(random.uniform(0,1),5)
      headActivationS = round(random.uniform(0,1),5)

      leftWristActivationO = round(random.uniform(0,1),5)
      leftWristActivationS = round(random.uniform(0,1),5)
      
      rightWristActivationO = round(random.uniform(0,1),5)
      rightWristActivationS = round(random.uniform(0,1),5)

      waistActivationO = round(random.uniform(0,1),5)
      waistActivationS = round(random.uniform(0,1),5)

      leftAnkleActivationO = round(random.uniform(0,1),5)
      leftAnkleActivationS = round(random.uniform(0,1),5)
      
      rightAnkleActivationO = round(random.uniform(0,1),5)
      rightAnkleActivationS = round(random.uniform(0,1),5)      

      
#Sending of Point Coordinates to TouchDesigner

      # Send Head Coordinates through Port 5001
      
      parser = argparse.ArgumentParser()
      parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
      parser.add_argument("--port", type=float, default=PORT1, help="The port the OSC server is listening on")
      args = parser.parse_args()
      client = udp_client.SimpleUDPClient(args.ip, args.port)

      client.send_message("/head", headX)
      client.send_message("/head", headY)
      
      # Send Wrists Coordinates through Port 5002

      parser = argparse.ArgumentParser()
      parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
      parser.add_argument("--port", type=float, default=PORT2, help="The port the OSC server is listening on")
      args = parser.parse_args()
      client = udp_client.SimpleUDPClient(args.ip, args.port)

      client.send_message("/leftWristX", leftWristX)
      client.send_message("/leftWristY", leftWristY)

      client.send_message("/rightWristX", rightWristX)
      client.send_message("/rightWristY", rightWristY)
      print(leftWristX)

      # Send Waist Coordinates through Port 5003

      parser = argparse.ArgumentParser()
      parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
      parser.add_argument("--port", type=float, default=PORT3, help="The port the OSC server is listening on")
      args = parser.parse_args()
      client = udp_client.SimpleUDPClient(args.ip, args.port)

      client.send_message("/waist", waistX)
      client.send_message("/waist", waistY)
      
      # Send Ankles Coordinates through Port 5004

      parser = argparse.ArgumentParser()
      parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
      parser.add_argument("--port", type=float, default=PORT4, help="The port the OSC server is listening on")
      args = parser.parse_args()
      client = udp_client.SimpleUDPClient(args.ip, args.port)

      client.send_message("/leftAnkleX", leftAnkleX)
      client.send_message("/leftAnkleY", leftAnkleY)

      client.send_message("/rightAnkleX", rightAnkleX)
      client.send_message("/rightAnkleY", rightAnkleY)


#Sending of Point Coordinates to TouchDesigner

      # Send package with all Activation Values through Port 5100
      
      parser = argparse.ArgumentParser()
      parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
      parser.add_argument("--port", type=float, default=PORT5, help="The port the OSC server is listening on")
      args = parser.parse_args()
      client = udp_client.SimpleUDPClient(args.ip, args.port)

      client.send_message("/headActivationO", headActivationO)
      client.send_message("/headActivationS", headActivationS)
      client.send_message("/leftWristActivationO", leftWristActivationO)
      client.send_message("/leftWristActivationS", leftWristActivationS)
      client.send_message("/rightWristActivationO", rightWristActivationO)
      client.send_message("/rightWristActivationS", rightWristActivationS)
      client.send_message("/waistActivationO", waistActivationO)
      client.send_message("/waistActivationS", waistActivationS)
      client.send_message("/leftAnkleActivationO", leftAnkleActivationO)
      client.send_message("/leftAnkleActivationS", leftAnkleActivationS)
      client.send_message("/rightAnkleActivationO", rightAnkleActivationO)
      client.send_message("/rightAnkleActivationS", rightAnkleActivationS)



    frame_id += 1     

    # Display Camera Strea - Flip image horizontally for selfie-view display.
    img = cv2.resize(image, (900,1000), interpolation=cv2.INTER_AREA)
    cv2.imshow('MediaPipe Pose', cv2.flip(img, 1))  
    cv2.resizeWindow('MediaPipe Pose',900,1000)    

    if cv2.waitKey(10) & 0xFF  == ord('q'): #presionar q para cerrar la camara
      break
cap.release()
