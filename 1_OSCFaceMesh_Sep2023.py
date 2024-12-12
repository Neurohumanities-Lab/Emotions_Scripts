import cv2
import mediapipe as mp
import numpy as np
from pythonosc import udp_client
import argparse
import time



flagMarkers = 0 # 1 to show Markers



IP = "10.12.181.191"  # IP Address for TouchDesigner
PORT1 = 10001  #Left Eye X
PORT2 = 10002  #Left Eye Y
PORT3 = 10003  #Right Eye X
PORT4 = 10004  #Right Eye Y
PORT5 = 10005  #Nose Tip X and Y
PORT6 = 10006  #Left Face Edge X and Y
PORT7 = 10007  #Right Face Edge X and Y
PORT8 = 10008  #Bottom Face Edge X and Y



################################################################
##Sides are Inverted Compared to Standard online Documentation## 
################################################################

LeftEyeIDs = [33, 159, 133, 145] #4 Markers for Left Eye
RightEyeIDs = [362, 386, 263, 374] #4 Markers for Right Eye
NoseTipIDs = [1] #1 Marker for Nose Tip
LeftFaceEdgeIDs = [93] #1 Marker for Left Face Edge 
RightFaceEdgeIDs = [323] #1 Marker for Right Face Edge
BottomFaceEdgeIDs = [200] #1 Marker for Bottom Face Edge



print("-------------------------------------------")
print("OSC Transfer of Face Marker Coordinates.")
print("Displaying Camera Input for TouchDesigner's ScreenGrab.")
print("")
print("Sending OSC messages on IP " + str(IP) + " and ports " + str(PORT1) + ", " + str(PORT2)  + ", " + str(PORT3)  + ", " + str(PORT4)  + ", " + str(PORT5)  + ", " + str(PORT6)  + ", " + str(PORT7) + ", and " + str(PORT8) +".")
print("")
print("Stream includes:")
print(PORT1)
print("4 Rounded Floats with 5 decimals.")
print("X Coordinates for Left Eye Markers: ",  LeftEyeIDs)
print("")
print(PORT2)
print("4 Rounded Floats with 5 decimals.")
print("Y Coordinates for Left Eye Markers: ",  LeftEyeIDs)
print("")
print(PORT3)
print("4 Rounded Floats with 5 decimals.")
print("X Coordinates for Right Eye Markers: ",  RightEyeIDs)
print("")
print(PORT4)
print("4 Rounded Floats with 5 decimals.")
print("Y Coordinates for Right Eye Markers: ",  RightEyeIDs)
print("")
print(PORT5)
print("2 Rounded Floats with 5 decimals.")
print("X and Y Coordinates for Nose Tip Marker: ",  NoseTipIDs)
print("")
print(PORT6)
print("2 Rounded Floats with 5 decimals.")
print("X and Y Coordinates for Left Face Edge Marker: ",  LeftFaceEdgeIDs)
print("")
print(PORT7)
print("2 Rounded Floats with 5 decimals.")
print("X and Y Coordinates for Right Face Edge Marker: ",  RightFaceEdgeIDs)
print("")
print(PORT8)
print("2 Rounded Floats with 5 decimals.")
print("X and Y Coordinates for Bottom Face Edge Marker: ",  BottomFaceEdgeIDs)
print("")
print("Refresh Rate depends on internal FaceMesh processing speed - not hardcoded into main lines of code.")
print("-------------------------------------------")
print("Developed by Jesús Tamez-Duque and Alexandro Ortiz.")
print("-------------------------------------------")

counter = 0

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.moveWindow('Video',3850,0) 
cv2.resizeWindow('Video',1300,800)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if flagMarkers == 0:
            cv2.imshow("Video", frame) # Display Video without Markers 

        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks is not None:

          
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1))
        

            #Send Values to OSC

            # Left Eye X #

            parser = argparse.ArgumentParser()
            parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
            parser.add_argument("--port", type=float, default=PORT1, help="The port the OSC server is listening on (1)")
            args = parser.parse_args()

            client = udp_client.SimpleUDPClient(args.ip, args.port)

            for ID in LeftEyeIDs:
                client.send_message("/LeftEye.MarkerID"+str({ID})+".X", round(float(face_landmarks.landmark[ID].x),5))




            # Left Eye Y #

            parser = argparse.ArgumentParser()
            parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
            parser.add_argument("--port", type=float, default=PORT2, help="The port the OSC server is listening on (1)")
            args = parser.parse_args()

            client = udp_client.SimpleUDPClient(args.ip, args.port)

            for ID in LeftEyeIDs:
                client.send_message("/LeftEye.MarkerID"+str({ID})+".Y", round(float(face_landmarks.landmark[ID].y),5))




            # Right Eye X #

            parser = argparse.ArgumentParser()
            parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
            parser.add_argument("--port", type=float, default=PORT3, help="The port the OSC server is listening on (1)")
            args = parser.parse_args()

            client = udp_client.SimpleUDPClient(args.ip, args.port)

            for ID in RightEyeIDs:
                client.send_message("/RightEye.MarkerID"+str({ID})+".X", round(float(face_landmarks.landmark[ID].x),5))




            # Right Eye Y #
        
            parser = argparse.ArgumentParser()
            parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
            parser.add_argument("--port", type=float, default=PORT4, help="The port the OSC server is listening on (1)")
            args = parser.parse_args()

            client = udp_client.SimpleUDPClient(args.ip, args.port)

            for ID in RightEyeIDs:
                client.send_message("/RightEye.MarkerID"+str({ID})+".Y", round(float(face_landmarks.landmark[ID].y),5))




            # Nose Tip X and Y #
        
            parser = argparse.ArgumentParser()
            parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
            parser.add_argument("--port", type=float, default=PORT5, help="The port the OSC server is listening on (1)")
            args = parser.parse_args()

            client = udp_client.SimpleUDPClient(args.ip, args.port)

            for ID in NoseTipIDs:
                client.send_message("/NoseTip.MarkerID"+str({ID})+".X", round(float(face_landmarks.landmark[ID].x),5))
                client.send_message("/NoseTip.MarkerID"+str({ID})+".Y", round(float(face_landmarks.landmark[ID].y),5))




            # Left Face Edge X and Y #
        
            parser = argparse.ArgumentParser()
            parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
            parser.add_argument("--port", type=float, default=PORT6, help="The port the OSC server is listening on (1)")
            args = parser.parse_args()

            client = udp_client.SimpleUDPClient(args.ip, args.port)

            for ID in LeftFaceEdgeIDs:
                client.send_message("/LeftFaceEdge.MarkerID"+str({ID})+".X", round(float(face_landmarks.landmark[ID].x),5))
                client.send_message("/LeftFaceEdge.MarkerID"+str({ID})+".Y", round(float(face_landmarks.landmark[ID].y),5))




            # Right Face Edge X and Y #
        
            parser = argparse.ArgumentParser()
            parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
            parser.add_argument("--port", type=float, default=PORT7, help="The port the OSC server is listening on (1)")
            args = parser.parse_args()

            client = udp_client.SimpleUDPClient(args.ip, args.port)

            for ID in RightFaceEdgeIDs:
                client.send_message("/RightFaceEdge.MarkerID"+str({ID})+".X", round(float(face_landmarks.landmark[ID].x),5))
                client.send_message("/RightFaceEdge.MarkerID"+str({ID})+".Y", round(float(face_landmarks.landmark[ID].y),5))




            # Bottom Face Edge X and Y #
        
            parser = argparse.ArgumentParser()
            parser.add_argument("--ip", default=IP, help="The ip of the OSC server")
            parser.add_argument("--port", type=float, default=PORT8, help="The port the OSC server is listening on (1)")
            args = parser.parse_args()

            client = udp_client.SimpleUDPClient(args.ip, args.port)

            for ID in BottomFaceEdgeIDs:
                client.send_message("/BottomFaceEdge.MarkerID"+str({ID})+".X", round(float(face_landmarks.landmark[ID].x),5))
                client.send_message("/BottomFaceEdge.MarkerID"+str({ID})+".Y", round(float(face_landmarks.landmark[ID].y),5))

            client = udp_client.SimpleUDPClient(args.ip, 5003)
            dist = np.sqrt(np.power(face_landmarks.landmark[13].x - face_landmarks.landmark[14].x,2) + np.power(face_landmarks.landmark[13].y - face_landmarks.landmark[14].y,2))
            client.send_message("/mouth", round(float(dist),5))
            time.sleep(0.1)

            # Display Video with Markers #

            if flagMarkers == 1:
                cv2.imshow("FaceMesh with Markers", frame)
        else:
            counter += 1
            print('NO FACE is detected: '+ str(counter) + ' times')
            

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        

cap.release()
cv2.destroyAllWindows()
