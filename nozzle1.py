import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import config as cfg

# OBS: ao selecionar a região da cor do elemento a ser detectado, fazer a média da cor do objeto.
# com esta média, rastrear por esta cor.
# hue (matiz), saturation (saturação) e value (valor) = (HSV = [0-360 =qual cor],[0-100 = saturaçao] ,[0-100=brilho])
# encontrar a porcentagem de cada cor, utilizando a saturação do HSV

def plotaComGraph (v1, threshIm):
    titles = ['Original','Thresh']
    images = [v1, threshIm]
    for i in range(len(images)):
        plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

def bgrDivisor(img):
    b,g,r = cv2.split(img)

    cv2.imshow("blue",b)
    cv2.imshow("green",g)
    cv2.imshow("red",r)

def rgbDivisor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bgrImage = cv2.split(image)

    b = bgrImage[2].reshape(image.shape[0],image.shape[1])
    g = bgrImage[1].reshape(image.shape[0],image.shape[1])
    r = bgrImage[0].reshape(image.shape[0],image.shape[1])
    fig, ax = plt.subplots(3)
    ax[0].imshow(r, cmap="Reds")  
    ax[1].imshow(g, cmap="Greens")
    ax[2].imshow(b, cmap="Blues")
    plt.show()

def calcAverageColorToDetect(image):

    np.array(image)
    colorAverage = []
    r, g, b, count = 0, 0, 0, 0
    for line in image:
        for pixel in line:
            r += pixel[0]
            g += pixel[1]
            b += pixel[2]
            count +=1
    print(int(round(r/count)), int(round(g/count)),int(round (b/count)))
    
def calibrateGraphcapture(image):
    r = cv2.selectROI(image)
    imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.imshow("ROI",imCrop)
    
    return imCrop

def hsvDivisor(imgage):
    hsv = cv2.cvtColor(imgage, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return h, s, v, hsv

def autoCanny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def __printLine(image, line):
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        #print("x0,y0", x0, y0)
        #print("y1:", y1, "y2", y2)
        cv2.line(image, (x1,y1), (x2,y2), cfg.PURE_RED, cfg.DRAW_LINES_THICKNES)

def findLinesOnImage(image, edges):
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    linesToMeasure = {}
    height = image.shape[0]
    indexes = []

    for line in lines:
        linesToMeasure[int(line[0][0])] = line
        indexes.append(int(line[0][0]))
    
    lineIndexTop = min(indexes, key=lambda x:abs(x- int(height/2)))
    lineIndexBottom = max(indexes, key=lambda x:abs(x- int(height/2)))
    __printLine(image, linesToMeasure[lineIndexTop])
    __printLine(image, linesToMeasure[lineIndexBottom])
    
    return lineIndexTop, lineIndexBottom
     
def validateAreaAndContours(contours, lineIndexTop, lineIndexBottom):
    contoursList = []
    areaList = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        #print("contorno: ",contour)
        if (contour[0][0][1] > lineIndexTop 
                and contour[0][0][1] < lineIndexBottom
                and ((len(approx) > 1) & (area > cfg.AREA_BALLS_SENSIBILITY))):
            contoursList.append(contour)
            areaList.append(area)
    return contoursList, areaList

def findCenterCordinates(contoursList):
    centerCordinates = []
    for contour in range (len(contoursList)):
        M = cv2.moments(contoursList[contour])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centerCordinates.append([cX, cY])
    return centerCordinates

def reproduceCleanScenary(image, centerCordinates, startMeasure=0, endMeasure=0):
    height, width, depth = image.shape
    ptosVetor = centerCordinates

    circlesImage = np.zeros((height, width), np.uint8)

    for pto in range (len(ptosVetor)):
        cv2.circle(circlesImage, (ptosVetor[pto][0], ptosVetor[pto][1]), 4, (255, 255, 255), -1) # PRINT A CIRCLE on cordinates
        
    return circlesImage

def calculatePercentMeasureLevel(image, centerCordinates):
    height, width, _ = image.shape
    for pointCenter in range (len(centerCordinates)):
        percent = ((centerCordinates[pointCenter][1] - height ) / height)
        print ("Tubo -",pointCenter," ", math.fabs(percent)*100, "%")
        perAnt = percent
        xArrayPrevious = centerCordinates[pointCenter][0]
    
def kernelProcess(image, sigma=100):
    
    imageCopy = image.copy()  # Copia um frame em outra array
    blur = (3,3)
    
    # aplica o blur
    imageBlur = cv2.blur(imageCopy, blur)
    
    # converte a imagem em cinza
    grayBluredImage = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray-blur', grayBluredImage)

    # dilata a imagem das bordas
    kernelDilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    openingImage = cv2.dilate(grayBluredImage, kernelDilate, iterations = 3)
    #cv2.imshow('Opening', openingImage)

    # detecta bordas
    # comprime a imagem das bordas
    kernelSize = 3
    kernelErosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    erodeImage = cv2.erode(grayBluredImage, kernelErosion, iterations = 3)
    #cv2.imshow('Erosion', erodeImage)

    # detecta bordas
    kernelSize = cfg.KERNEL_SIZE_CANNY
    closingDetectedEdges = cv2.Canny(erodeImage, cfg.CANNY_DOWN_THRESHOLD, cfg.CANNY_UPPER_THRESHOLD, kernelSize)
    cv2.imshow('edges-Open', closingDetectedEdges)
    
    lineIndexTop, lineIndexBottom = findLinesOnImage(image, closingDetectedEdges)

    contours, _ = cv2.findContours(closingDetectedEdges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    contoursList, _ = validateAreaAndContours(contours, lineIndexTop, lineIndexBottom)
    
    centerCordinates = findCenterCordinates(contoursList)
    cv2.drawContours(image, contoursList, -1, cfg.PURE_GREEN, cfg.DRAW_LINES_THICKNES)         
    
    
    circleImage = reproduceCleanScenary(image, centerCordinates, startMeasure=lineIndexBottom, endMeasure=lineIndexBottom)
    newCleanedImage = circleImage[lineIndexTop:lineIndexBottom]
    #############
    contours, _ = cv2.findContours(newCleanedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    contoursList, _ = validateAreaAndContours(contours, lineIndexTop, lineIndexBottom)
    centerCordinates = findCenterCordinates(contoursList)
    calculatePercentMeasureLevel(image, centerCordinates)
    #############

    cv2.imshow("circulos", circleImage)
    cv2.imshow("image", image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def printFrameInfo(cap):
    high, width, channels = cap.shape
    print("Altura:",high)
    print("Largura:",width)
    print("Cannels:", channels)

def nuzzleMeasure():
    cap = cv2.imread('/Users/hz/Documents/nuzzleGraph/fotos/tubos.jpg')    
    if cap is None:
        exit()
    cv2.imshow("ORIGINAL", cap)
    kernelProcess(cap)
    
if __name__ == "__main__":
    nuzzleMeasure()



