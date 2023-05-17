import cv2
import mediapipe as mp

import pyautogui
# from pynput.keyboard import Key,Controller
from pynput.mouse import Button,Controller


import math

virtualmouse = Controller()
pinch = False

video = cv2.VideoCapture(0)


width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

screenwidth,screenheight = pyautogui.size()
print (screenwidth, screenheight)

print(width,height)

myhands=mp.solutions.hands

# print("my hands: ", myhands)
mydrawing=mp.solutions.drawing_utils

hand_object=myhands.Hands(min_detection_confidence=0.75,min_tracking_confidence=0.75) 

# print("what is hand_object: ", hand_object)

def count_fingers(myimage,lst):
    count=0
    global pinch

    thresh = (lst.landmark[0].y*100-lst.landmark[9].y*100)/2

    if (lst.landmark[5].y*100 - lst.landmark[8].y*100)>thresh:
        count+=1
    if (lst.landmark[9].y*100 - lst.landmark[12].y*100)>thresh:
        count+=1
    if (lst.landmark[13].y*100 - lst.landmark[16].y*100)>thresh:
        count+=1
    if (lst.landmark[17].y*100 - lst.landmark[20].y*100)>thresh:
        count+=1

    tf = count
    
    finger_tip_x = int(lst.landmark[8].x*width)
    finger_tip_y = int(lst.landmark[8].y*height)
    
    thumb_tip_x = int(lst.landmark[4].x*width)
    thumb_tip_y = int(lst.landmark[4].y*height)

    cv2.line(myimage, (finger_tip_x, finger_tip_y),(thumb_tip_x,thumb_tip_y),(0,255,145),2)

    centerx = int((finger_tip_x + thumb_tip_x)/2)
    centery = int((finger_tip_y + thumb_tip_y)/2)
    cv2.circle(myimage,(centerx,centery),2,(145,165,125),2)
    distance = math.sqrt(((finger_tip_x - thumb_tip_x)**2)+(finger_tip_y - thumb_tip_y)**2)
    relative_mouse_x = (centerx/width)*screenwidth
    relative_mouse_y = (centery/height)*screenheight
    virtualmouse.position = (relative_mouse_x, relative_mouse_y)
    print(virtualmouse.position)
    if distance>40:
        if pinch == True:
            pinch = False
            virtualmouse.release(Button.left)
    
    if distance<40:
        if pinch == False:
            pinch = True
            virtualmouse.press(Button.left)



    return tf


while True:
    dummy,frame = video.read()
    flipImage=cv2.flip(frame,1)

    result=hand_object.process(cv2.cvtColor (flipImage,cv2.COLOR_BGR2RGB)) 
    # print(result.multi_hand_landmarks)

    if result.multi_hand_landmark:
        hand_keypoints = result.multi_hand_landmarks[0]
        mydrawing.draw_landmarks(flipImage,hand_keypoints,myhands.HAND_CONECTIONS)
        
        mycount = count_fingers(flipImage,hand_keypoints)
        cv2.putText(flipImage,"count" +str(mycount),(100,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,225),2)

    cv2.imshow('Hand Guestures: ',flipImage)

    if cv2.waitKey(25) == 32:
        break
    

video.release()
cv2.destroyAllWindows()