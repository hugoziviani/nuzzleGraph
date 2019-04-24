import numpy as np
import cv2


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,0,10])
    upper_black = np.array([255,255,0])

    mask = cv2.inRange(hsv, lower_black, upper_black)

    cv2.imshow('Alterado', mask)
    # Display the resulting frame
    #cv2.threshold(frame, 200, THRESH_BINARY, THRESH_BINARY)->retval, frame
    cv2.imshow('FRAME',gray)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
