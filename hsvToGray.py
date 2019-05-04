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

cap = cv2.imread('/Users/hz/Documents/nozzlePy/tubos2.png')
high, width, channels = cap.shape

print("Altura:",high)
print("Largura:",width)
# desenho p1 (0,0) p2(raio+i,

frame = cap
hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
h, s, v1 = cv2.split(hsv)

blur = cv2.medianBlur(v1,7)
cv2.imshow('Blur', blur)

lower_black = np.array([0])
upper_black = np.array([35])
mask = cv2.inRange(blur, lower_black, upper_black)
ret2, threshIm = cv2.threshold(mask, 34, 255,cv2.THRESH_BINARY)


kernelO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
opening = cv2.morphologyEx(threshIm, cv2.MORPH_OPEN, kernelO)
#cv2.imshow('Opening', opening)


kernelE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
erosion = cv2.erode(opening,kernelE,iterations = 3)
#cv2.imshow('Erosion', erosion)

kernelC = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
dilationC = cv2.dilate(opening,kernelO,iterations = 2)
#cv2.imshow('DilatadaC', dilationC)

kernelO = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
dilation = cv2.dilate(opening,kernelO,iterations = 4)
cv2.imshow('Dilatada', dilation)

contornos = dilation
bilateral_filtered_image = cv2.bilateralFilter(contornos, 5, 175, 175)
edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
contours, a1 = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#contours - é um vetor com os contornos
#o contorno é um vetor com os pontos
contour_list = []
area_list = []


for contour in contours:
    approx = cv2.approxPolyDP(contour,0.010*cv2.arcLength(contour,True),True)
    area = cv2.contourArea(contour)
    #print("contorno: ",contour)
    if ((len(approx) > 8) & (area > 30)):
        area_list.append(area)
        contour_list.append(contour)

# varrer os contornos selecionados buscando a menor e a maior area de contorno.
print ("Area tamanho:",len(area_list))
print ("Contornos tamanho:",len(contour_list))

#print(contours)
longeDis = 0.0
pertoDis = 0.0
distA = 0
distB = 0




i = 0
for contorno in contour_list:
    xP = 0
    yP = 0
    xL = 0
    yL = 0
    for pto in range (len(contorno)):
        xA = 0
        yA = 0
        xB = contorno.item(pto,0,0)
        yB = contorno.item(pto,0,1)

        distAB = math.sqrt(((xA-xB)**2) + ((yA-yB)**2))
        print (distAB)
        if(pto==0):
            longeDis = distAB
            pertoDis = distAB
        if(pertoDis < distAB):
            pertoDis = distAB
            xP = int(contorno.item(pto,0,0))
            yP = int(contorno.item(pto,0,1))
            print("xP:",xP,"yP",yP)
            
        if(longeDis > distAB):
            longeDis = distAB
            xL = int(contorno.item(pto,0,0))
            yL = int(contorno.item(pto,0,1))
            print("xL:",xP,"yL",yP)
    
        cv2.rectangle(frame,(xP,0),(xL, high),(0,255,0),1)

    print(i)
    i = i+1


#varrer os contornos
#pegar o pto mais distante e o mais próximo da origem



# media das areas para cortar em setores a figura
#setor = math.sqrt(maiorArea/math.pi)
#enaquato houver foto ele vai dividir pelo tamanho do setor.


cv2.drawContours(frame, contour_list, -1, (0,255,0), 1)
cv2.imshow("comContornos", frame)






#Criar um ponto auxiliar que recebe o maior e o menor do contorno e setar na lista para ser desenhada as linhas
#cv2.line(frame,(menorP[0],menorP[1]),(maiorP[0],maiorP[1]),(255,0,0),1)

                        #print (distAB)



#print("Menor: ",menorP)
#print("Maior: ",maiorP)
#cv2.line(frame, menorP,  -1, (255,0,0), 1)
#cv2.line(frame, maiorD,  -1, (0,0,255), 1)






#for area in area_list:
#print(math.sqrt(float(area))*3.14)
#cv2.imshow('Objects Detected', frame)

#for tubo in range(61):
#cv2.rectangle(frame,(0,0),(int(x)*int(tubo), h),(0,255,0),1)
#cv2.imshow('Objects Detected', frame)


#cv2.imshow('gray!', v1)
#cv2.imshow('Thrash!', threshIm)
cv2.waitKey(0)
cv2.destroyAllWindows()


