import os
import numpy as np
import cv2 as cv
from sklearn.svm import SVC

def loadData(path, is_norm = True):
    
    dict = {'cuan': '川', 'e1': '鄂', 'gan': '赣', 'gan1': '甘', 'gui': '贵', 'gui1': '桂', \
    'hei': '黑', 'hu': '沪', 'ji': '冀', 'jin': '津', 'jing': '京', 'jl': '吉', 'liao': '辽', \
        'lu': '鲁', 'meng': '蒙', 'min': '闽', 'ning': '宁', 'qing': '青', 'qiong': '琼', \
        'shan': '陕', 'su': '苏', 'sx': '晋', 'wan': '皖', 'xiang': '湘', 'xin': '新', 'yu': '豫', \
        'yu1': '渝', 'yue': '粤', 'yun': '云', 'zang': '藏', 'zhe': '浙'}
    
    train_X = []
    test_X = []
    train_Y = []
    test_Y = []
    cates = os.listdir(path)
    for cate in cates:
        if cate == '.DS_Store': continue
        imgsPath = os.path.join(path, cate)
        imgs = os.listdir(imgsPath)
        length = len(imgs)
        id = 0
        for img in imgs:
            if img == '.DS_Store': continue
            imgPath = os.path.join(imgsPath, img)
            # print(imgPath)
            image = cv.resize(cv.imread(imgPath, 0), (20, 20), interpolation=cv.INTER_CUBIC).reshape(-1,)
            id += 1
            if (id > length * 0.85): 
                test_X.append(image)
                if cate not in dict:
                    test_Y.append(cate)
                else: test_Y.append(dict[cate])
            else:
                train_X.append(image)
                if cate not in dict:
                    train_Y.append(cate)
                else: train_Y.append(dict[cate])
    
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    if is_norm:
        train_X = (train_X - np.mean(train_X, axis= 0)) / (np.std(train_X, axis= 0))
        test_X = (test_X - np.mean(test_X, axis= 0)) / (np.std(test_X, axis= 0))
    train_Y = np.array(train_Y)
    test_Y = np.array(test_Y)
    
    # print(len(train_Y))
    # print(len(test_Y))
    
    return train_X, test_X, train_Y, test_Y
    
if __name__ == '__main__':
    train_X, test_X, train_Y, test_Y = loadData('./MyData', is_norm= False)
    svm = SVC()
    # print(train_X.shape)
    # print(train_Y.shape)
    svm.fit(train_X, train_Y)
    
    test_point = cv.resize(cv.imread('./characters/6.png', 0), (20, 20), interpolation=cv.INTER_CUBIC).reshape(1, -1)
    # test_point = (test_point - np.mean(test_point, axis= 0)) / (np.std(test_point, axis= 0))
    # cv.imwrite('./t.png', test_point.reshape(20, 20))
    print(svm.predict(test_point))
    
    # predict_Y = svm.predict(test_X)
    # print(predict_Y)
    # print(test_Y)
    # print((predict_Y == test_Y).sum() / len(test_Y))