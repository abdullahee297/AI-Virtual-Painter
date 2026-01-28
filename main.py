import cv2
import time 
import os
import numpy as np 
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisualRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = "hand_landmarker.task"),
    running_mode = VisualRunningMode.IMAGE,
    num_hands = 2
)

detector = HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                  # Thumb
    (0, 5), (5, 9), (9, 13), (13, 17), (17, 0),      # Wist
    (5, 6), (6, 7), (7, 8),                          # Index
    (9, 10), (10, 11), (11, 12),                     # Middle
    (13, 14), (14, 15), (15, 16),                    # Ring
    (17, 18), (18, 19), (19, 20)                     # Pinky
]

# Importing the images for the overlay

folderPath = "header_img"               # Folder Path
myList = os.listdir(folderPath)         # Accessing the folder
# print(myList)                           # Show the files in the folder
overlay = []                            # Variable that store all the images

# One by one import the image and save it in the array
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlay.append(image)
# print(len(overlay))                     # Length / Number of images in the array

header = overlay[0]
resize_header = None
drawcolor = (0, 255, 0)
brushthickness = 15
eraserThickness = 40
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while True:
    success, img = cap.read()
    
    if not success:
        break

    # Set the header image at the top
    h, w, _ = img.shape

    # To do only one time to optimize
    if resize_header is None:
        resize_header = cv2.resize(header, (w, 125))

    img[0:125, 0:w] = resize_header

    finger_count = 0


    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(mp.ImageFormat.SRGB,rgb)
    
    result = detector.detect(mp_img)
    
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            h, w, _ = img.shape
            lm_list = []

            for lm in hand:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            if len(lm_list) >= 21:
                x1, y1 = lm_list[8]             # Index finger tip c0ordinates
                x2, y2 = lm_list[12]            # Middle finger tip coordinates


            else:
                continue  
            
            # print(x1, y1)              

            for start, end in HAND_CONNECTIONS:
                cv2.line(img, lm_list[start], lm_list[end], (0, 255, 0), 2)
            
            for x, y in lm_list:
                cv2.circle(img, (x, y), 6, (255, 0, 0), -1)


            # Finger Count (Up/Down)
            finger = []
            if lm_list[4][0] > lm_list[3][0]:
                finger_count += 1
                finger.append(1)
            else:
                finger.append(0)

            # Other fingers (y-axis check)
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]

            for tip, pip in zip(tips, pips):
                if lm_list[tip][1] < lm_list[pip][1]:
                    finger_count += 1
                    finger.append(1)
                else:
                    finger.append(0)

            #print(finger)

            # Making the mode Condition
            if finger[1] and finger[2]:
                cv2.rectangle(img, (x1, y1 - 35), (x2, y2 + 35), (255, 0, 255), cv2.FILLED)
                # print("Selection Mode")
                # Checking for the click
                if y1 < 125:
                    # For the first one 250 - 360
                    if 250 < x1 < 400:
                        header = overlay[1]             #green
                        drawcolor = (0 , 255, 0)
                        resize_header = None
                    # For the Second one 450 - 570
                    elif 450 < x1 < 650:
                        header = overlay[2]             #red
                        drawcolor = (0 , 0, 255)
                        resize_header = None
                    # For the Third one 690 - 890
                    elif 690 < x1 < 900:
                        header = overlay[3]             #blue
                        drawcolor = (255, 0, 0)
                        resize_header = None
                    # For the Fourth one 
                    elif 930 < x1 < 1050:
                        header = overlay[4]             #yellow
                        drawcolor = (0, 255, 255)
                        resize_header = None
                    # For the Fifth one 
                    elif 1090 < x1 < 1250:
                        header = overlay[5]             #eraser
                        drawcolor = (0, 0, 0)
                        resize_header = None
                cv2.rectangle(img, (x1, y1 - 35), (x2, y2 + 35), drawcolor, cv2.FILLED)
                xp, yp = x1, y1

            if finger[1] and not finger[2]:
                cv2.circle(img, (x1, y1), 20, drawcolor, cv2.FILLED)
                # print("Drawing Mode")
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawcolor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawcolor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushthickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, brushthickness)
                xp, yp = x1, y1
            

    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("AI Virtual Painter", img)
    # cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()