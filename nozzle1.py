import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #gray = cv2.cvtColor(hsv, cv2.COLOR_HSV2GRAY)
    ret2, trashIm = cv2.threshold(hsv,127,255,cv2.THRESH_BINARY)

    lower_black = np.array([0,0,0])
    upper_black = np.array([0,0,10])
    mask = cv2.inRange(trashIm, lower_black, upper_black)
    
    
    # Display the resulting frame
    #cv2.imshow('cinza',gray)
    #cv2.imshow('original',frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
