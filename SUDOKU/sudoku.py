
import joblib
import numpy as np
from numpy import logical_and as land
from numpy import logical_not as lnot
from skimage.feature import canny
from skimage.transform import rescale, ProjectiveTransform, warp
from skimage.morphology import dilation, disk
from matplotlib import pyplot as plt
import cv2
from skimage.measure import label
from skimage.feature import match_template
from skimage import io
import os

temp1 = cv2.imread('/autograder/submission/template1.jpg')
temp2 = cv2.imread('/autograder/submission/template2.jpg')
temp3 = cv2.imread('/autograder/submission/template3.jpg')
temp4 = cv2.imread('/autograder/submission/template4.jpg')
temp5 = cv2.imread('/autograder/submission/template5.jpg')
temp6 = cv2.imread('/autograder/submission/template6.jpg')
temp7 = cv2.imread('/autograder/submission/template7.jpg')
temp8 = cv2.imread('/autograder/submission/template8.jpg')
temp9 = cv2.imread('/autograder/submission/template9.jpg')

temps = [temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9]

temps = [cv2.resize(temp, (50,50)) for temp in temps]

class SUDOKU:
    def __init__(self, img):
        self.image = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.blurGain = 3
        self.kernalSize = 9
        self.nunScanOrder = [8, 5, 9, 4, 1, 2, 7, 3, 6 ]
        self.thresh = [0.7, 0.62 , 0.58, 0.675, 0.58, 0.675, 0.75 ,0.62 ,0.6 ]

        self.digits = np.zeros([9,9]) -1
        self.h = 550
        self.w = 550

    def preprocessImg(self):
        grey = self.gray
        blur_gain = self.blurGain
        blur = cv2.GaussianBlur(grey,(blur_gain,blur_gain),0)
        th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
        ret3,th = cv2.threshold(th,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_c = 255-th.astype(np.uint8)
        k = self.kernalSize
        kernel = np.ones((k,k))
        img_c = cv2.morphologyEx(img_c, cv2.MORPH_CLOSE, kernel)
        return img_c


    def get_mask (self):
        img_ = self.preprocessImg()
        contours, hierarchy = cv2.findContours(img_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        index = []
        i = 0
        for cnt in contours:
            areas.append(cv2.contourArea(cnt))
            index.append(i)
            i+=1
        area_dict = dict(zip(index, areas))
        areas = np.sort(areas)
        def get_key(d, value):
            for k, v in d.items():
                if v == value:
                    return k
        mask = np.zeros((self.image.shape[0],self.image.shape[1]))
        mask = cv2.fillPoly(mask, [contours[get_key(area_dict, areas[-1])]], 255)
        return mask/255


    def normalized (self):
        img_ = self.preprocessImg()
        contours, hierarchy = cv2.findContours(img_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        index = []
        i = 0
        for cnt in contours:
            areas.append(cv2.contourArea(cnt))
            index.append(i)
            i+=1
        area_dict = dict(zip(index, areas))
        areas = np.sort(areas)
        def get_key(d, value):
            for k, v in d.items():
                if v == value:
                    return k
        rect = cv2.minAreaRect(contours[get_key(area_dict, areas[-1])])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        perimeter = cv2.arcLength(contours[get_key(area_dict, areas[-1])], True) 
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(contours[get_key(area_dict, areas[-1])],epsilon,True)
        if approx.shape[0] == 4:
            box = approx.reshape(4,-1)
        rect = np.zeros((4, 2), dtype = "float32")
        s = box.sum(axis = 1)
        rect[0] = box[np.argmin(s)]
        rect[2] = box[np.argmax(s)]
        diff = np.diff(box, axis = 1)
        rect[1] = box[np.argmin(diff)]
        rect[3] = box[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.image, M, (maxWidth, maxHeight))
        return warped

    def get_cell(self, binary_img, row, column, delta=25):
        w,h = binary_img.shape[:2]
        h_s, w_s = round(h / 9), round(w / 9)
        #print(image[y-delta : y + delta, x - delta : x + delta].shape)
        return binary_img[w_s*row : w_s*(row+1), h_s*column : h_s*(column+1)]

    def get_local_centers(self, corr, th):
        lbl, n = label(corr , connectivity=2, return_num=True)
        return np.int16([np.round(np.mean(np.argwhere(lbl == i), axis=0)) for i in range(1, n + 1)])

    def point2cell(self, points, num, d=2):
        if num == 1:
            d = 10
        else:
            d = 2
        for p in range (len(points)):
            for i in range (9):
                for j in range (9):
                    if points[p,0] >= i*self.h/9 + d and points[p,0] < (i+1)*self.h/9 - d\
                        and points[p,1] >= j*self.w/9 + d and points[p,1] < (j+1)*self.w/9 - d:
                        self.digits[i,j] = num

    def write_digit(self, num, th):
        wrappedImg = self.normalized()
        k = 1
        kernel = np.ones((k,k))
        wrappedImg = cv2.morphologyEx(wrappedImg, cv2.MORPH_CLOSE, kernel)
        wrapped_gray = cv2.cvtColor(wrappedImg.copy() , cv2.COLOR_BGR2GRAY)
        wrapped_gray = cv2.resize(wrapped_gray, (self.h, self.w))
        binary_img = wrapped_gray < 80
        assert num>0 and num<10 and num/np.int(num) ==1
        temp = temps[num-1]
        res = match_template(wrapped_gray, temp , pad_input = True )
        points = self.get_local_centers(res>th, 0.5)
        self.point2cell(points, num)

    def get_digit(self):
        for num in self.nunScanOrder:
            th = self.thresh[num-1]
            self.write_digit(num, th)
        return self.digits



def predict_image(image: np.ndarray):
    sudoku = SUDOKU(image)
    digits = sudoku.get_digit()
    mask = sudoku.get_mask()
    return mask, digits
