
# Computer Vision -> HW-1 TICKET TO RIDE
# Nipun Weerakkodi
# Skoltec, MSc 2022
# 25/11/2021


import cv2
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from skimage.feature import match_template
from skimage.measure import label
import math
from scipy.signal import convolve2d


# ================================================================================================
# ================================================================================================

PATH1 = ''  # when not running on autograder
PATH2 = '/autograder/submission/'   # if upload to autograder '/autograder/submission/'
PATH = PATH2
print('path changed')

if np.any(cv2.imread(f'{PATH}red_track.jpg') == None):
    PATH = PATH1

# path test:
check = cv2.imread(f'{PATH}red_track.jpg')
if np.any(check == None):
    print('ERROR: PATH to template images is wrong!!!')
    print('Please select the correct path for template images!!!')
    quit()
    

class train_colors:
    def __init__ (self, img):
        self.img = img
        self.tools = predict_tools(img)
        self.rgbImg = self.tools.rgbImg
        self.hsvImg = self.tools.hsvImg             
        self.temp_bTrack = cv2.imread(f'{PATH}blue_track_tracker.jpg')
        self.temp_gTrack = cv2.imread(f'{PATH}green_track_tracker.jpg')
        self.temp_yTrack = cv2.imread(f'{PATH}yellow_track_tracker.jpg')
        self.temp_rTrack = cv2.imread(f'{PATH}red_track.jpg')
    #-----------------------------------------------------------
    def count_blueTrains(self, plot_cnts= False, l_range = np.array([100, 150,110 ]), \
        u_range = np.array([120, 255,180 ]), debug_plots = False ):

        HSL_lower =l_range
        HSL_upper = u_range
        mask_ = cv2.inRange(self.hsvImg.copy(), HSL_lower, HSL_upper)
        mask = cv2.medianBlur(mask_, 3)
        mask_int = mask.astype(np.uint8)
        kernel = np.ones((1,1))
        mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel)
        cnts, hierarchy = cv2.findContours(mask_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
        #remove repeated, inside, small contours
        cnts = self.contour_filter(cnts, hierarchy, 1600)
        id = []
        if len(cnts) == 0:
            noOf_blue_trains = 0
            return noOf_blue_trains
        else:
            #remove contours of blue tracks with no trains
            b_tracks = self.get_blue_tracks(plot=False)
            b_tracks_ = b_tracks - [5,15]
            # getting tracks id
            for i in range (len(b_tracks_)):
                for j in range(len(cnts)):
                    x_cnts = cnts[j][:,0,1]
                    y_cnts = cnts[j][:,0,0]
                    if b_tracks_[i, 0] >= min(x_cnts) and b_tracks_[i, 0] <= max(x_cnts)\
                        and b_tracks_[i, 1] >= min(y_cnts) and b_tracks_[i, 1] <= max(y_cnts):
                        id.append(j)
            blue =  cnts
            #remove tracks
            id.sort(reverse=True)
            for i in id:
                cnts.pop(i)

            attached_cnts = self.attached_contours(cnts)
            if len(cnts) > 0:
                for i in range (len(cnts)):
                    if cv2.contourArea(cnts[i]) > 5500 and cv2.contourArea(cnts[i]) < 2*5000:
                        attached_cnts +=1
                    if cv2.contourArea(cnts[i]) >  2*5000 and cv2.contourArea(cnts[i]) < 3*5000:
                        attached_cnts +=2  
            #plot
            if plot_cnts == True:
                plt.figure(figsize=(20,10))
                rgb = self.rgbImg.copy()
                cv2.drawContours(rgb, cnts, -1, (0,255,0), 3)
                plt.imshow(rgb)
                plt.show()

            if debug_plots == True:
                fig, ax = plt.subplots(2,2, figsize=(50,25))
                ax[0,0].imshow(mask)
                ax[0,0].set(title='mask')
                ax[0,1].imshow(mask_int)
                ax[0,1].set(title='mask_int')
                rgb1 = self.rgbImg.copy()
                cv2.drawContours(rgb1, cnts, -1, (0,255,0), 3)
                ax[1,0].imshow(rgb1)
                ax[1,0].set(title='blue trains')
                rgb2 = self.rgbImg.copy()
                cv2.drawContours(rgb2, blue, -1, (0,255,0), 3)
                ax[1,1].imshow(rgb2)
                ax[1,1].set(title='all blue')
                plt.show()

            n_blue_trains = len(cnts) + attached_cnts

            return n_blue_trains
    #-----------------------------------------------------------
    def count_greenTrains(self, plot_cnts= False, l_range = np.array([75, 160,50 ]), \
        u_range = np.array([83, 255,145 ]), debug_plots = False ):

        HSL_lower =l_range
        HSL_upper = u_range
        mask = cv2.inRange(self.hsvImg.copy(), HSL_lower, HSL_upper)
        mask_int = mask.astype(np.uint8)
        kernel = np.ones((7,7))
        mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel)
        cnts, hierarchy = cv2.findContours(mask_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
        #remove repeated, inside, and small contours
        cnts1 = self.contour_filter(cnts, hierarchy, 700)
        if len(cnts1) == 0:
            noOf_green_trains = 0
            return noOf_green_trains
        else:
            #count attached train contours
            attached_cnts= self.attached_contours(cnts1, area=4800)
            
            #plot
            if plot_cnts == True:
                plt.figure(figsize=(20,10))
                rgb = self.rgbImg.copy()
                cv2.drawContours(rgb, cnts1, -1, (0,255,0), 3)
                plt.imshow(rgb)
                plt.show()

            if debug_plots == True:
                fig, ax = plt.subplots(2,2, figsize=(50,25))
                ax[0,0].imshow(mask)
                ax[0,0].set(title='mask')
                ax[0,1].imshow(mask_int)
                ax[0,1].set(title='mask_int')
                rgb1 = self.rgbImg.copy()
                cv2.drawContours(rgb1, cnts1, -1, (0,255,0), 3)
                ax[1,0].imshow(rgb1)
                ax[1,0].set(title='green trains')
                plt.show()

            n_green_trains = len(cnts1) + attached_cnts 

            return n_green_trains
    #-----------------------------------------------------------
    def count_blackTrains(self, plot_cnts = False, plot_mask= False):
        Img = self.img
        img_blur = cv2.GaussianBlur(Img,(7,5),0)
        Img_ = img_blur[..., ::-1]
        HLS = cv2.cvtColor(Img_, cv2.COLOR_RGB2HLS)
        HUE = HLS[:, :, 0]              # Split attributes
        LIGHT = HLS[:, :, 1]
        SAT = HLS[:, :, 2]
        
        mask_black = (LIGHT < 25)
        mask_int_black = mask_black.astype(np.uint8)
        kernel = np.ones((15,15))
        mask_int_black = cv2.morphologyEx(mask_int_black, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((19,19))
        mask_int_black = cv2.morphologyEx(mask_int_black, cv2.MORPH_OPEN, kernel)
        if plot_mask == True:
            plt.figure(dpi=200)
            plt.imshow(mask_int_black)
        cnts, hierarchy = cv2.findContours(mask_int_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = self.contour_filter(cnts, hierarchy, 4000 )
        rgb_im1 = cv2.cvtColor(Img.copy(), cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb_im1, cnts, -1, (0,255, 0), 2)
        if plot_cnts == True:
            plt.figure(dpi=200)
            plt.imshow(rgb_im1)
        attached = self.attached_contours(cnts, 3514)
        # print( len(cnts) , attached)
        return len(cnts) + attached
    #-----------------------------------------------------------    
    def count_yellowTrains(self, plot_cnts = False, l_range = np.array([21, 74,180 ]), \
        u_range = np.array([37, 225,255 ]), debug_plots = False ):

        HSL_lower =l_range
        HSL_upper = u_range
        mask = cv2.inRange(self.hsvImg.copy(), HSL_lower, HSL_upper)
        mask_int = mask.astype(np.uint8)
        k = 3
        kernel = np.ones((k,k))
        mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel)
        cnts, hierarchy = cv2.findContours(mask_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE ) 
        #remove repeated, inside, small contours
        cnts = self.contour_filter(cnts, hierarchy, 2000)
        all_y = cnts.copy()
        if len(cnts) == 0:
            noOf_yellow_trains = 0
            return noOf_yellow_trains
        else:
            #remove contours of yellow tracks with no trains
            y_tracks_ = self.get_yellow_tracks(plot=debug_plots)
            y_tracks = y_tracks_ 
            id = []
            for i in range (len(y_tracks)):
                for j in range(len(cnts)):
                    x_cnts = cnts[j][:,0,1]
                    y_cnts = cnts[j][:,0,0]
                    if y_tracks[i, 0] >= min(x_cnts) and y_tracks[i, 0] <= max(x_cnts)\
                        and y_tracks[i, 1] >= min(y_cnts) and y_tracks[i, 1] <= max(y_cnts):
                        id.append(j)
            #remove tracks
            id.sort(reverse=True)

            for i in id:
                cnts.pop(i)


            attached_cnts = self.attached_contours(cnts)
            #plot
            if plot_cnts == True:
                plt.figure(figsize=(20,10))
                rgb = self.rgbImg.copy()
                cv2.drawContours(rgb, cnts, -1, (0,255,0), 3)
                plt.imshow(rgb)
                plt.show()

            if debug_plots == True:
                fig, ax = plt.subplots(2,2, figsize=(50,25))
                ax[0,0].imshow(mask)
                ax[0,0].set(title='mask')
                ax[0,1].imshow(mask_int)
                ax[0,1].set(title='mask_int')
                rgb1 = self.rgbImg.copy()
                cv2.drawContours(rgb1, cnts, -1, (0,255,0), 3)
                ax[1,0].imshow(rgb1)
                ax[1,0].set(title='yellow trains')
                rgb2 = self.rgbImg.copy()
                cv2.drawContours(rgb2, all_y , -1, (0,255,0), 3)
                ax[1,1].imshow(rgb2)
                ax[1,1].set(title='all Yellow')
                plt.show()

            n_yellow_trains = len(cnts) + attached_cnts

            return n_yellow_trains
    #-----------------------------------------------------------
    def count_redTrains(self, plot_cnts = False, plot_mask = False):
        Img = self.img
        img_blur = cv2.GaussianBlur(Img,(9,5),0)
        Img_ = img_blur[..., ::-1]
        HLS = cv2.cvtColor(Img_, cv2.COLOR_RGB2HLS)
        HUE = HLS[:, :, 0]              # Split attributes
        LIGHT = HLS[:, :, 1]
        SAT = HLS[:, :, 2]

        mask_red = ( (HUE > 160)  ) & (SAT > 120) &  (SAT < 210) #& (LIGHT > 150)
        mask_int_red = mask_red.astype(np.uint8)
        k = 9
        kernel = np.ones((k,k))
        mask_int_red = cv2.morphologyEx(mask_int_red, cv2.MORPH_OPEN, kernel)
        mask_int_red = cv2.medianBlur(mask_int_red,5)
        if np.any(plot_cnts or plot_mask):
            plt.figure(dpi=200)
        if plot_mask == True:
            plt.subplot(1,2,1)
            plt.imshow(mask_int_red)
        cnts, hierarchy = cv2.findContours(mask_int_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = self.contour_filter(cnts, hierarchy, 800 )
        rgb_im1 = cv2.cvtColor(Img.copy(), cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb_im1, cnts, -1, (0,255, 0), 2)
        if plot_cnts == True:
            plt.subplot(1,2,2)
            plt.imshow(rgb_im1)
        return len(cnts) + self.attached_contours(cnts, 5800)
    #-----------------------------------------------------------
    # ==============================================================================================
    def contour_filter(self, contours, hierarchy, minArea):
        cnts1 = []
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1 and cv2.contourArea(contours[i]) > minArea:
                cnts1.append(contours[i])
        return cnts1
    #-----------------------------------------------------------
    def attached_contours(self, cnts, area = 5000):
        attached_cnts = 0
        if len(cnts) > 0:
            for i in range (len(cnts)):
                if cv2.contourArea(cnts[i]) >= area and cv2.contourArea(cnts[i]) < 2*area:
                    attached_cnts +=1
                if cv2.contourArea(cnts[i]) >=  2*area and cv2.contourArea(cnts[i]) < 3*area:
                    attached_cnts +=2 
                if cv2.contourArea(cnts[i]) >= 4*area and cv2.contourArea(cnts[i]) < 5*area:
                    attached_cnts +=3
                if cv2.contourArea(cnts[i]) >=  5*area and cv2.contourArea(cnts[i]) < 6*area:
                    attached_cnts +=4 
        return attached_cnts
    #-----------------------------------------------------------
    def get_blue_tracks(self, plot = False):
        bTracks = self.tools.find_sameObjects(self.temp_bTrack, thresh = 0.5, blur=1, plot=plot)
        return bTracks
    #-----------------------------------------------------------
    def get_green_tracks(self, plot=False):
        gTracks = self.tools.find_sameObjects(self.temp_gTrack, thresh = 0.455, blur=1, plot=True)
        return gTracks
    #-----------------------------------------------------------
    def get_yellow_tracks(self, plot=False):
        yTracks = self.tools.find_sameObjects(self.temp_yTrack, thresh = 0.55 , blur=1, plot=plot)
        return yTracks
    def get_red_tracks(self, plot=False):
        rTracks = self.tools.find_sameObjects(self.temp_rTrack, thresh = 0.55 , blur=1, plot=plot)
        return rTracks
    #-----------------------------------------------------------

# ================================================================================================
# ================================================================================================


class centers:
    def __init__(self, img):
        self.img = img
        self.tools = predict_tools(img)
        self.points = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []

        self.template1 = cv2.imread(f'{PATH}center1.jpg')
        self.template2 = cv2.imread(f'{PATH}center2.jpg')
        self.template3 = cv2.imread(f'{PATH}center3.jpg')
        self.template4 = cv2.imread(f'{PATH}center4.jpg')
    #-----------------------------------------------------------
    def get_center_points(self, plot = False):
        p1 = self.tools.find_sameObjects(self.template1, thresh = 0.84 , blur=1, plot=False)
        p2 = self.tools.find_sameObjects(self.template2, thresh = 0.84 , blur=1, plot=False)
        p3 = self.tools.find_sameObjects(self.template3, thresh = 0.8 , blur=1, plot=False)
        p4 = self.tools.find_sameObjects(self.template4, thresh = 0.84 , blur=1, plot=False)
        # print('p1 found: ', len(p1))
        self.p1 = p1 + [5, 0]
        self.p2 = p2 + [-20,0]
        self.p3 = p3 + [0,-20]
        self.p4 = p4 + [-20,-20]
        self.points = self.tools.filterPoints(self.tools.filterPoints(self.tools.filterPoints(p1, self.p2) , self.p3), self.p4)
        if plot ==True:
            self.tools.plot_rect(self.points, self.template1.shape ) 
        return self.points
    #-----------------------------------------------------------

# ================================================================================================
# ================================================================================================


class predict_tools:
    def __init__ (self, img):
        self.img = img
        self.rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #-----------------------------------------------------------
    def plot_img(self, img, cmap='gray', figsize = (20,10)):
        plt.figure(figsize=figsize)
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        plt.show()
    #-----------------------------------------------------------
    def plot_rect(self, points,  bbox_shape ): 
        rgb = self.rgbImg.copy()
        points = np.int16(points)[::, ::-1]
        res_img = np.int16(rgb.copy())
        for pt in points:
            cv2.rectangle(res_img, (pt[0] - bbox_shape[0] // 2, pt[1] - bbox_shape[1] // 2),
                        (pt[0] + bbox_shape[0] // 2, pt[1] + bbox_shape[1] // 2), (255, 0, 0), 3)
        self.plot_img(res_img, cmap=None, figsize=(40,20))
    #-----------------------------------------------------------
    # for debug 
    def plot_points(self, points, p = 0):
        
        d = 20
        self.plot_img(self.img[int(points[p,0]-d):int(points[p,0]+d), int(points[p,1]-d):(points[p,1]+d)])
    #-----------------------------------------------------------
    def get_local_centers(self, corr, th):
        lbl, n = label(corr >= th, connectivity=2, return_num=True)
        return np.int16([np.round(np.mean(np.argwhere(lbl == i), axis=0)) for i in range(1, n + 1)])
    #-----------------------------------------------------------
    def find_sameObjects(self, template, thresh = 0.779, blur = 3, plot=False):
        grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        grayBlurImg = cv2.medianBlur(grayImg, blur)
        grayTemp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        grayblurTemp = cv2.medianBlur(grayTemp, blur)
        # plt.imshow(grayblurTemp, cmap='gray')
        corr_skimage = match_template(grayBlurImg, grayblurTemp, pad_input=True)
        points = self.get_local_centers(corr_skimage>thresh, 0.5)   # 0.779 for img1, img2   /   for img3 0.7738
        if plot == True:   
            self.plot_rect(points, template.shape)
        return points
    #-----------------------------------------------------------
    def filterPoints(self, p1, p2):
        # getting the list of centers using points get from templates 
        for i in range (len(p1)):
            # print('i=',i)
            j = 0
            while True:
                if len(p2) == 0:
                    break
                elif self.p2p_dist(p1[i], p2[j]):
                    p2 = np.delete(p2, j, 0 ) # if p2 has the same center, delete it from p2
                if j >= len(p2)-1 :
                    break
                else: 
                    j +=1

        points = np.append(p1, p2, axis=0)
        points = np.int16(np.asarray(points))
        return points
    #-----------------------------------------------------------
    def p2p_dist(self, p1, p2, thresh = 50, value = False):
        if value:
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return dist
        else:
            if math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < thresh :
                # print(dist_)
                return True
            else:
                return False
    #-----------------------------------------------------------

# ================================================================================================
# ================================================================================================


# ===========================================================================

def green_score(img):
    blur_ = 1
    k1 = 1
    k2 = 9
    #------------------------------------
    Img = cv2.GaussianBlur(img,(blur_,blur_),0)
    Img = Img[..., ::-1]
    HLS = cv2.cvtColor(Img, cv2.COLOR_BGR2HLS)
    l_range = np.array([29, 0, 103])
    u_range = np.array([61, 77, 191])
    #red mask
    mask_red = cv2.inRange(HLS, l_range, u_range)
    mask_int_red = mask_red.astype(np.uint8)
    kernel = np.ones((k2,k2))
    mask_int_red = cv2.morphologyEx(mask_int_red, cv2.MORPH_CLOSE, kernel)
    #Red conturs
    mask_int_red = cv2.medianBlur(mask_int_red,k2)
    contours, hierarchy = cv2.findContours(mask_int_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = train_colors.contour_filter(None, contours, hierarchy, 4000)
    areas = [cv2.contourArea(cnt) for cnt in cnts]
    areas.sort()
    area_th = 5400             # max 6767
    score = 0
    for a in areas:
        if a < area_th:
            score+=1
        elif a>=area_th and a < 2*area_th:
            score +=2
        elif a>=2*area_th and a<3*area_th:
            score += 4
        elif a>=3*area_th and a<4*area_th:
            score += 7
        elif a>=4*area_th and a<5*area_th:
            score += 15
        else:
            score +=21
    return score

def blue_score(img):
    blur_ = 1
    k1 = 1
    k2 = 11
    image = img
    rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB )
    Img = cv2.GaussianBlur(image,(blur_,blur_),0)
    Img = Img[..., ::-1]
    #------------------------------------
    HLS = cv2.cvtColor(Img, cv2.COLOR_BGR2HLS)
    HUE = HLS[:, :, 0]              # Split attributes
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]

    l_range = np.array([19, 0, 96])
    u_range = np.array([27, 96, 216])
    #red mask
    mask_red = cv2.inRange(HLS, l_range, u_range)
    mask_int_red = mask_red.astype(np.uint8)
    kernel = np.ones((k2,k2))
    mask_int_red = cv2.morphologyEx(mask_int_red, cv2.MORPH_CLOSE, kernel)
    #Red conturs
    mask_int_red = cv2.medianBlur(mask_int_red,k2)
    contours, hierarchy = cv2.findContours(mask_int_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = train_colors.contour_filter(None, contours, hierarchy, 2000)

    areas = [cv2.contourArea(cnt) for cnt in cnts]
    areas.sort()


    area_th = 4220         # max 6767
    score = 0
    for a in areas:
        if a < area_th:
            score+=1
        elif a>=area_th and a < 2*area_th:
            score +=2
        elif a>=2*area_th and a<3*area_th:
            score += 4
        elif a>=3*area_th and a<4*area_th:
            score += 7
        elif a>=4*area_th and a<5*area_th:
            score += 15
        else:
            score +=21

    # print(areas)
    return score

def red_score(img):
    blur_ = 1
    k1 = 1
    k2 = 15
    image = img
    rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB )
    Img = cv2.GaussianBlur(image,(blur_,blur_),0)
    Img = Img[..., ::-1]
    #------------------------------------
    HLS = cv2.cvtColor(Img, cv2.COLOR_BGR2HLS)
    HUE = HLS[:, :, 0]              # Split attributes
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]

    l_range = np.array([121, 86, 148])
    u_range = np.array([132, 148, 190])
    #red mask
    mask_red = cv2.inRange(HLS, l_range, u_range)
    mask_int_red = mask_red.astype(np.uint8)
    kernel = np.ones((k2,k2))
    mask_int_red = cv2.morphologyEx(mask_int_red, cv2.MORPH_CLOSE, kernel)
    #Red conturs
    mask_int_red = cv2.medianBlur(mask_int_red,k2)
    contours, hierarchy = cv2.findContours(mask_int_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = train_colors.contour_filter(None, contours, hierarchy, 2000)

    areas = [cv2.contourArea(cnt) for cnt in cnts]
    areas.sort()

    area_th = 4000         # max 6767
    score = 0
    for a in areas:
        if a < area_th:
            score+=1
        elif a>=area_th and a < 2*area_th:
            score +=2
        elif a>=2*area_th and a<3*area_th:
            score += 4
        elif a>=3*area_th and a<4*area_th:
            score += 7
        elif a>=4*area_th and a<5*area_th:
            score += 15
        else:
            score +=21

    # print(areas)
    return score

def black_score(img):

    blur_ = 1
    k1 = 1
    k2 = 15
    image = img
    rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB )
    Img = cv2.GaussianBlur(image,(blur_,blur_),0)
    Img = Img[..., ::-1]
    #------------------------------------
    HLS = cv2.cvtColor(Img, cv2.COLOR_BGR2HLS)
    HUE = HLS[:, :, 0]              # Split attributes
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]

    # l_range = np.array([121, 86, 148])
    # u_range = np.array([132, 148, 190])
    # #red mask
    # mask_red = cv2.inRange(HLS, l_range, u_range)

    mask_red = (LIGHT < 25)
    mask_int_red = mask_red.astype(np.uint8)
    kernel = np.ones((k2,k2))
    mask_int_red = cv2.morphologyEx(mask_int_red, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((19,19))
    mask_int_black = cv2.morphologyEx(mask_int_red, cv2.MORPH_OPEN, kernel)
    #Red conturs
    mask_int_red = cv2.medianBlur(mask_int_red,k2)
    contours, hierarchy = cv2.findContours(mask_int_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = train_colors.contour_filter(None, contours, hierarchy, 5000)

    areas = [cv2.contourArea(cnt) for cnt in cnts]
    areas.sort()

    area_th = 9200      # max 6767
    score = 0
    for a in areas:
        if a < area_th:
            score+=1
        elif a>=area_th and a < 2*area_th:
            score +=2
        elif a>=2*area_th and a<3*area_th:
            score += 4
        elif a>=3*area_th and a<4*area_th:
            score += 7
        elif a>=4*area_th and a<5*area_th:
            score += 15
        else:
            score +=21
    return score

def yellow_score(img):
    blur_ = 5
    k1 = 1
    k2 = 9
    image = img
    rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB )
    Img = cv2.GaussianBlur(image,(blur_,blur_),0)
    Img = Img[..., ::-1]
    #------------------------------------
    HLS = cv2.cvtColor(Img, cv2.COLOR_BGR2HLS)
    HUE = HLS[:, :, 0]              # Split attributes
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]

    l_range = np.array([65, 96, 103])
    u_range = np.array([105, 157, 191])
    #red mask
    mask_red = cv2.inRange(HLS, l_range, u_range)

    # mask_red = (LIGHT < 25)
    mask_int_red = mask_red.astype(np.uint8)
    kernel = np.ones((k2,k2))
    mask_int_red = cv2.morphologyEx(mask_int_red, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((19,19))
    # mask_int_black = cv2.morphologyEx(mask_int_red, cv2.MORPH_OPEN, kernel)
    #Red conturs
    mask_int_red = cv2.medianBlur(mask_int_red,k2)
    contours, hierarchy = cv2.findContours(mask_int_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = train_colors.contour_filter(None,contours, hierarchy, 3000)

    areas = [cv2.contourArea(cnt) for cnt in cnts]
    areas.sort()

    area_th = 6700        # max 6767
    score = 0
    for a in areas:
        if a < area_th:
            score+=1
        elif a>=area_th and a < 2*area_th:
            score +=2
        elif a>=2*area_th and a<3*area_th:
            score += 4
        elif a>=3*area_th and a<4*area_th:
            score += 7
        elif a>=4*area_th and a<5*area_th:
            score += 15
        else:
            score +=21
    return score




def predict_image(img):

    score_b = 0
    score_g = 0
    score_k = 0
    score_y = 0
    score_r = 0
    print('Image scanning........\n')
    
    c = centers(img.copy())
    center_p = c.get_center_points(plot = False)
    print('Centers found.\n')
    print('Scanning trains and getting score\n')
    t = train_colors(img.copy())
    n_blue      = t.count_blueTrains()
    if n_blue > 0 :
        score_b = blue_score(img.copy())
    n_green     = t.count_greenTrains()
    if n_green> 0 :
        score_g = green_score(img.copy())
    n_black     = t.count_blackTrains()
    if n_black > 0 :
        score_k = black_score(img.copy())
    n_yellow    = t.count_yellowTrains()
    if n_yellow > 0 :
        score_y = yellow_score(img.copy())
    n_red       = t.count_redTrains()
    if n_blue > 0 :
        score_r = red_score(img.copy())
    n_trains = {'blue': n_blue, 'green': n_green, 'black': n_black, 'yellow': n_yellow, 'red': n_red}
    score = {'blue': score_b, 'green': score_g, 'black': score_k, 'yellow': score_y, 'red':score_r}
    center_p = np.int64(center_p)

    return center_p, n_trains, score

    
