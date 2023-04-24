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
        y1, y2, y3, y4 = int(y1), int(y2), int(y3), int(y4)
        while imageMask[int(rect[0][1]), x1] == 0: x1 += 1
        while imageMask[int(rect[0][1]), x3] == 0: x3 -= 1
        box = makeBOX([x1, y1], [x2, y2], [x3, y3], [x4, y4])

        boxes.append(np.intp(box))
    
    
    imageRGB = cv.drawContours(imageRGB, boxes, -1, (0, 0, 255), 5)
    cv.imwrite("test5.png", imageRGB)

    return boxes
    
def split_plate(boxes, imageRGB):
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
        cv.imwrite("test6.png", warped)

        imageList.append(warped)
    return imageList


if __name__ == '__main__':
    image_path = "./images/difficult/3-2.jpg"
    imageRGB = cv.imread(image_path)
    cv.imwrite("test.png", imageRGB)
    contourBoxes = find_plate(imageRGB.copy())
    plates = split_plate(contourBoxes, imageRGB.copy())
    print(len(plates))