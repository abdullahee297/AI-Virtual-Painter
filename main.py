import cv2
import time 
import os
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

    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(mp.ImageFormat.SRGB,rgb)
    
    result = detector.detect(mp_img)
    
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            h, w, _ = img.shape
            lm_list = []

            if lm_list != 0:
                print(lm_list)
            
            for lm in hand:
                lm_list.append((int(lm.x*w), int(lm.y*h)))

            for start, end in HAND_CONNECTIONS:
                cv2.line(img, lm_list[start], lm_list[end], (0, 255, 0), 2)
            
            for x, y in lm_list:
                cv2.circle(img, (x, y), 6, (255, 0, 0), -1)

            # if result.handedness:
            #     for i, hand_info in enumerate(result.handedness):
            #         label = hand_info[0].category_name   # "Left" or "Right"
            #         score = hand_info[0].score

            # print(f"Hand {i}: {label}")
                
    
    cv2.imshow("AI Virtual Painter", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()