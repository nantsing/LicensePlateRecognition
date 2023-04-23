import math
import cv2 as cv
import numpy as np

def makeBOX(p1, p2, p3, p4):
    return np.array([np.array(p1),
                     np.array(p2),
                     np.array(p3),
                     np.array(p4),
                     ])

def find_plate(imageRGB) -> np.array :
    imageHSV = cv.cvtColor(imageRGB, cv.COLOR_BGR2HSV)
    imageHSV = cv.GaussianBlur(imageHSV, (7, 7), 0)
    cv.imwrite("test1.png", imageHSV)

    ######## blue ########
    blue_low = np.array([100, 115, 115])
    blue_high = np.array([124, 255, 255])

    ######## green ########
    green_low = np.array([35, 10, 160])
    green_high = np.array([70, 100, 200])

    imageMask = cv.inRange(imageHSV, blue_low, blue_high) +\
         cv.inRange(imageHSV, green_low, green_high)

    cv.imwrite("test2.png", imageMask)

    ######## Morphological Processing ########
    kernel = np.ones((5, 5), np.uint8)
    imageMask = cv.morphologyEx(imageMask, cv.MORPH_OPEN, kernel)
    cv.imwrite("test3.png", imageMask)
    kernel = np.ones((40, 40), np.uint8)
    imageMask = cv.morphologyEx(imageMask, cv.MORPH_CLOSE, kernel)
    cv.imwrite("test4.png", imageMask)

    ######## Find Contours ########
    MinArea = 3000
    contourCandidates, _ = cv.findContours(imageMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contourCandidates = [candidate for candidate in contourCandidates if cv.contourArea(candidate) > MinArea]
    # print(len(contourCandidates))
    # print(contourCandidates)
    boxes = []
    for candidate in contourCandidates:
        rect = cv.minAreaRect(candidate)
        width, height = rect[1]
        theta = rect[2]
        if width < height:
            width, height = height, width
            
        if width / height < 2.5 or width / height > 5.0:
            continue
        # print(width/height)
        box = cv.boxPoints(rect)
        
        ######## left or right ########
        if theta > 45:
            dy = height / math.sin(math.radians(theta))
        else :
            dy = height / math.cos(math.radians(theta))
        x2, y2 = box[1]
        x4, y4 = box[3]
        x1 = x4
        y1 = y4 - dy
        x3 = x2
        y3 = y2 + dy
        box = makeBOX([x1, y1], [x2, y2], [x3, y3], [x4, y4])
        # print(box)
        
        boxes.append(np.intp(box))
    
    # print(len(boxes))
    # print(boxes[0])
    imageMask = cv.drawContours(imageRGB, boxes, -1, (0, 0, 255), 5)
    cv.imwrite("test5.png", imageMask)

    
    

if __name__ == '__main__':
    image_path = "./images/difficult/3-2.jpg"
    imageRGB = cv.imread(image_path)
    cv.imwrite("test.png", imageRGB)
    find_plate(imageRGB)