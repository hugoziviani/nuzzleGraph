import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


def plotaComGraph (v1, threshIm):
    titles = ['Original','Thresh']
    images = [v1, threshIm]
    for i in range(len(images)):
        plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

#cap = cv2.VideoCapture(0)
cap = cv2.imread('/Users/hz/Documents/nuzzlePy/tubos2.png')
heigth, width, channels = img.shape
print(heigth)
print(width)

#while(True):

#ret, frame = cap.read()
frame = cap
hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)

#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
h, s, v1 = cv2.split(hsv)
bilateral_filtered_image = cv2.bilateralFilter(v1, 5, 175, 175)
#cv2.imshow('Bilateral', bilateral_filtered_image)
edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
#cv2.imshow('Edge', edge_detected_image)
contours, a1 = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

xP2 = []
contour_list = []
for contour in contours:
    approx = cv2.approxPolyDP(contour,0.025*cv2.arcLength(contour,True),True)
    area = cv2.contourArea(contour)
    if ((len(approx) > 8) & (area > 30) ):
        xP2 = float(area)/3.14
        contour_list.append(contour)
cv2.drawContours(frame, contour_list,  -1, (0,255,0), 1)

crop_img = []
y = 0
x = int(xP2)
h = 444
w = 10
for tubo in range(61):
    cv2.rectangle(frame,(0,0),(int(x)*int(tubo), h),(0,255,0),1)
    crop_img = v1[y:y+h*tubo, x:x+w]
cv2.imshow('Objects Detected', frame)
cv2.imshow('Croped!', crop_img)


lower_black = np.array([0])
upper_black = np.array([30])
mask = cv2.inRange(bilateral_filtered_image, lower_black, upper_black)
ret2, threshIm = cv2.threshold(mask, 50, 255,cv2.THRESH_BINARY)




cv2.imshow('trash', threshIm)
#cv2.imshow('gray!', v1)
#plotaComGraph(v1,threshIm)
#if cv2.waitKey(1) & 0xFF == ord('q'):
#   break
cv2.waitKey(0)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


