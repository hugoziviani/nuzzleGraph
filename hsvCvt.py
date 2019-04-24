import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    #cv2.threshold(frame, 200, THRESH_BINARY, THRESH_BINARY)->retval, frame

    #cv2.imshow('FRAME',gray)


    ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO_INV)

    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()




    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
