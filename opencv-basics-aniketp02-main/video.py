import argparse
import cv2 as cv
import numpy as np

# taking inputs from the terminal
parser = argparse.ArgumentParser()
parser.add_argument("file", help="path to image file")
parser.add_argument("width", nargs='?', type=int, help="width of the display screen")
parser.add_argument("height", nargs='?', type=int, help="height of the display screen")
args = parser.parse_args()

if args.width:
    w = args.width
else:
    w = 1280
if args.height:
    h = args.height
else:
    h = 720


file = args.file

# video feeds
cap = cv.VideoCapture(file)
cam_cap = cv.VideoCapture(0)

# mode to display the webcam default RGB(1)
mode = 1

while(True):

    ret, frame = cap.read()
    cam_ret, cam_frame = cam_cap.read()


    if ret:
        frame = cv.resize(frame, (w, h))

        # to draw the small x on ceter of screen
        cv.line(frame, (int(w/2) - 5,int(h/2) - 5), (int(w/2) + 5,int(h/2) + 5), (255,0,0), 2)
        cv.line(frame, (int(w/2) - 5,int(h/2) + 5), (int(w/2) + 5,int(h/2) - 5), (255,0,0), 2)
        cv.imshow("Image", frame)

    else:
       print('Looping the video')
       cap.set(cv.CAP_PROP_POS_FRAMES, 0)


    # red border for the webcam feed at the corner
    cam_frame = cv.copyMakeBorder(cam_frame, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=[0, 0, 255])
    cam_frame = cv.resize(cam_frame, (int(w/5), int(h/5)))


    if mode == 1:
        cv.imshow('Camera', cam_frame)
    elif mode == 2:
        cam_frame = cv.cvtColor(cam_frame, cv.COLOR_BGR2GRAY)
    elif mode == 3:
        cam_frame = cv.blur(cam_frame,(5,5))

    else:
        cv.imshow('Camera', cam_frame)


    cv.imshow('Camera', cam_frame)


    k = cv.waitKey(25)

    if k & 0xFF == ord('1'):
        mode = 1
    elif k & 0xFF == ord('2'):
        mode = 2
    elif k & 0xFF == ord('3'):
        mode = 3
    elif k & 0xFF == ord('Q'):
        break


cam_cap.release()
cap.release()
cv.destroyAllWindows()