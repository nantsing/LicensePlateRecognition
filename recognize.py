import os
import joblib
from load import *
from dataprocess import *
from sklearn.svm import SVC


def RecognizeCharacters(plate, model_path = './checkpoint/checpoint.pkl', data_path = './MyData'):
    svm = SVC(C= 1.0, kernel= 'rbf', decision_function_shape= 'ovr')
    if os.path.exists(model_path): svm = joblib.load(model_path)
    else:
        train_X, test_X, train_Y, test_Y = loadData(data_path, is_norm= False)
        svm.fit(train_X, train_Y)
        joblib.dump(svm, model_path)
        
    string = ''
    for idx, char in enumerate(plate):
        string = string + svm.predict(char)[0]
        if idx == 1:
            string = string + '·'
        
    return string

def RecognizePlates(plates, log = False):
    strings = []
    for plate in plates:
        SplitPlate = split_plate(plate, log= log)
        string = RecognizeCharacters(SplitPlate)
        strings.append(string)
        
    return strings

def Recognize(path, log = False):
    imageRGB = cv.imread(path)
    contourBoxes, imageCopy = find_plate(imageRGB.copy(), log= log)
    plates = cut_plates(contourBoxes, imageRGB.copy(), log= log)
    return imageCopy, plates, RecognizePlates(plates, log= log)

if __name__ == '__main__':
    # image_path = "./images/difficult/3-3.jpg"
    # image_path = "./images/medium/2-3.jpg"
    image_path = "./images/easy/1-3.jpg"
    imageRGB = cv.imread(image_path)
    plates = [imageRGB]
    # contourBoxes = find_plate(imageRGB.copy())
    # plates = cut_plates(contourBoxes, imageRGB.copy())
    print(RecognizePlates(plates, log= True))