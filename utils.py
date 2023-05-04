import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def ShowHisY(histogram):
    rows = len(histogram)
    plt.figure()
    plt.grid(True, linestyle=':', color='r', alpha=0.6)
    plt.barh(range(rows), histogram, height= 1, alpha= 0.6)
    plt.savefig('./fig/fig1.png')
    
def ShowHisX(histogram):
    rows = len(histogram)
    plt.figure()
    plt.grid(True, linestyle=':', color='r', alpha=0.6)
    plt.bar(range(rows), histogram, width= 1, alpha= 0.6)
    plt.savefig('./fig/fig2.png')

def Compute_his(plate, dim = 0):
    histgram = np.sum(plate, axis= dim)
    return histgram

def Cut_Y(histogram, threshold = 15000):
    rows = len(histogram)
    if histogram[0] > threshold: h1 = 0
    else:
        h1 = rows // 2 - 1 - np.argmin(histogram[: rows // 2][::-1])
    if histogram[rows - 1] > threshold: h2 = rows - 1
    else: 
        h2 = rows - 1 - np.argmin(histogram[rows // 2 :][::-1])
    
    return h1, h2

def Cut_X(histogram, threshold = 5000):
    X = []
    w1 = w2 = 0
    begin = False
    last = 0
    # cnt = 0
    rows = len(histogram)
    widthThreshold = rows // 15
    for i in range(rows):
        
        if histogram[i] == max(histogram) and i <= 10:
            last = histogram[i]
            continue
        
        if histogram[i] > threshold and last < threshold:
            begin = True
            last = histogram[i]
            w1 = i
            continue
        
        if (histogram[i] < threshold or i == rows - 1) and begin == True: 
            begin = False
            last = histogram[i]
            w2 = i
            if w2 - w1 < widthThreshold and histogram[(w1 + w2) // 2] < threshold * 20:
                continue
            X.append(np.array([w1, w2]))
            continue
            
        last = histogram[i]
        
    return np.array(X)