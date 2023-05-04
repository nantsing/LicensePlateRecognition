import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def makeBOX(p1, p2, p3, p4):
    return np.array([np.array(p1),
                     np.array(p2),
                     np.array(p3),
                     np.array(p4),
                     ])

def find_plate(imageRGB, log = False) -> np.array :
    imageHSV = cv.cvtColor(imageRGB, cv.COLOR_BGR2HSV)
    imageHSV = cv.GaussianBlur(imageHSV, (7, 7), 0)
    if log:
        cv.imwrite("./test/test1.png", imageHSV)

    ######## blue ########
    blue_low = np.array([100, 115, 115])
    blue_high = np.array([124, 255, 255])

    ######## green ########
    green_low = np.array([35, 10, 160])
    green_high = np.array([70, 100, 200])

    imageMask = cv.inRange(imageHSV, blue_low, blue_high) +\
         cv.inRange(imageHSV, green_low, green_high)
         
    if log:
        cv.imwrite("./test/test2.png", imageMask)

    ######## Morphological Processing ########
    kernel = np.ones((5, 5), np.uint8)
    imageMask = cv.morphologyEx(imageMask, cv.MORPH_OPEN, kernel)
    if log:
        cv.imwrite("./test/test3.png", imageMask)
    kernel = np.ones((35, 35), np.uint8)
    imageMask = cv.morphologyEx(imageMask, cv.MORPH_CLOSE, kernel)
    if log:
        cv.imwrite("./test/test4.png", imageMask)

    ######## Find Contours ########
    MinArea = 10000
    contourCandidates, _ = cv.findContours(imageMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contourCandidates = [candidate for candidate in contourCandidates if cv.contourArea(candidate) > MinArea]
    
    boxes = []
    for candidate in contourCandidates:
        rect = cv.minAreaRect(candidate)
        width, height = rect[1]
        theta = rect[2]
        if width < height:
            width, height = height, width
            
        if width / height < 2.5 or width / height > 5.0:
            continue
        box = cv.boxPoints(rect)
        
        ######## left or right ########
        if theta > 45:
            dy = height / math.sin(math.radians(theta))
            x2, y2 = box[1]
            x4, y4 = box[3]
            x1 = x4
            y1 = y4 - dy
            x3 = x2
            y3 = y2 + dy
        else :
            dy = height / math.cos(math.radians(theta))
            x1, y1 = box[1]
            x3, y3 = box[3]
            x2 = x3
            y2 = y3 - dy
            x4 = x1
            y4 = y1 + dy
        
        ######## fixes ########
        x1, x2, x3, x4 = int(x1), int(x2), int(x3), int(x4)
        y1, y2, y3, y4 = int(y1) + 8, int(y2) + 8, int(y3) - 16, int(y4) - 16
        while imageMask[int(rect[0][1]), x1] == 0: x1 += 1
        while imageMask[int(rect[0][1]), x3] == 0: x3 -= 1
        box = makeBOX([x1, y1], [x2, y2], [x3, y3], [x4, y4])

        boxes.append(np.intp(box))
    
    
    imageRGB = cv.drawContours(imageRGB, boxes, -1, (0, 0, 255), 5)
    if log:
        cv.imwrite("./test/test5.png", imageRGB)

    return boxes, imageRGB
    
def cut_plates(boxes, imageRGB, log = False):
    imageList = []
    for box in boxes:
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]

        # width = int(max( np.linalg.norm(np.array([x1 - x2, y1 - y2]), ord = 2), \
        #     np.linalg.norm(np.array([x3 - x4, y3 - y4]), ord = 2) ))
        height = max(abs(y1 - y4), abs(y2 - y3))
        width = int(3.2 * height)
        dst = np.array([[0, 0], [width - 1, 0], \
            [width - 1, height - 1], [0, height - 1]])
        box_ = np.array([[x1, y1], [x2, y2], \
            [x3, y3], [x4, y4]])
        
        M = cv.getPerspectiveTransform(box_.astype("float32"), dst.astype("float32"))
        warped = cv.warpPerspective(imageRGB, M, (width, height))
        if log:
            cv.imwrite("./test/test6.png", warped)

        imageList.append(warped)
    return imageList

######## blue or green ########
def split_plate(Plate, log = False):

    Plate = Plate[10: -10, :]
    Plate = Plate[:, 5: -5]
    
    plate = cv.GaussianBlur(Plate, (7, 7), 0)
    
    ######## blue ########
    blue_low = np.array([100, 115, 0])
    blue_high = np.array([180, 255, 255])

    ######## green ########
    green_low = np.array([0, 0, 100])
    green_high = np.array([180, 180, 255])
    
    ######## grey ########
    grey_low = np.array([0, 0, 0])
    grey_high = np.array([180, 190, 150])
    
    plate = cv.cvtColor(plate, cv.COLOR_BGR2HSV)
    
    testpoint = plate[plate.shape[0] // 2][5][0]
    print(testpoint)
    
    if testpoint < 90:
        plateMask = cv.inRange(plate, green_low, green_high) #+ cv.inRange(plate, grey_low, grey_high)
    else:
        plateMask = cv.inRange(plate, blue_low, blue_high) + cv.inRange(plate, grey_low, grey_high)
    
    height, width = plateMask.shape
    
    threshold_Y = height * width // 10
    threshold_X = height * width // 200
    
    cv.imwrite("./test/test6.png", plateMask)

    plateMask = np.where(plateMask == 0, 255, 0).astype('uint8')
    print(plateMask.shape)
    histogram = Compute_his(plateMask, 1)
    if log: ShowHisY(histogram)
    h1, h2 = Cut_Y(histogram, threshold_Y)
    
    plateMask = plateMask[h1:h2, :]
    # print(h1, h2)
    if log:
        cv.imwrite("./test/test7.png", plateMask)
    
    kernel = np.ones((3, 3), np.uint8)
    plateMask = cv.morphologyEx(plateMask, cv.MORPH_OPEN, kernel)
    kernel = np.ones((5, 5), np.uint8)
    plateMask = cv.morphologyEx(plateMask, cv.MORPH_CLOSE, kernel)
    if log:
        cv.imwrite("./test/test8.png", plateMask)
    
    k = plateMask.shape[0] // 14
    kernel = np.ones((k, k), np.uint8)
    dilate = cv.morphologyEx(plateMask, cv.MORPH_CLOSE, kernel)
    if log:
        cv.imwrite("./test/test9.png", dilate)
    
    histogram = Compute_his(dilate)
    if log: ShowHisX(histogram)
    SplitLines = Cut_X(histogram, threshold_X)
    plateMaskRGB = cv.cvtColor(plateMask, cv.COLOR_GRAY2RGB)
    SplitPlate = []
    for lines in SplitLines:
        w1, w2 = lines
        plateMaskRGB = cv.line(plateMaskRGB, (w1, 0), (w1, 1000), (0, 0, 255), 1)
        plateMaskRGB = cv.line(plateMaskRGB, (w2, 0), (w2, 1000), (0, 0, 255), 1)
        SplitPlate.append(cv.resize(plateMask[:, w1 : w2], (20, 20), interpolation=cv.INTER_CUBIC).reshape(1, -1)) 
    if log:
        cv.imwrite("./test/test10.png", plateMaskRGB)
    
    if log:
        for idx, char in enumerate(SplitPlate):
            cv.imwrite(f"./characters/{idx}.png", char.reshape(20, 20))
        
    return SplitPlate
        

if __name__ == '__main__':
    image_path = "./images/difficult/3-2.jpg"
    imageRGB = cv.imread(image_path)
    cv.imwrite("./test/test.png", imageRGB)
    contourBoxes, imageCopy = find_plate(imageRGB.copy(), log= True)
    plates = cut_plates(contourBoxes, imageRGB.copy(), log= True)
    # print(type(plates[0]))
    
    for plate in plates:
        SplitPlate = split_plate(plate, log= True)
    