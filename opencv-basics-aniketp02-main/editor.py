import argparse
import cv2 as cv
import numpy as np

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

print('Default color is blue')
color = (255, 0, 0) # color for the circle default blue

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),10,color,-1)


img = cv.imread(file)

if img is None:
    img = np.zeros((100, 100, 3))
    img.fill(255)

img = cv.resize(img, (w, h))


cv.namedWindow('Image')
cv.setMouseCallback('Image',draw_circle)


while(1):
    cv.imshow('Image',img)
    k = cv.waitKey(1)
    if k & 0xFF == ord('Q'):
        cv.imwrite('modified.jpg', img)
        break
    elif k & 0xFF == ord('R'):
        print('Current color selected is red')
        color = (0, 0, 255)
    elif k & 0xFF == ord('G'):
        print('Current color selected is green')
        color = (0, 255, 0)
    elif k & 0xFF == ord('B'):
        print('Current color selected is blue')
        color = (255, 0, 0)
        

cv.destroyAllWindows()