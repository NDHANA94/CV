import cv2
import numpy as np




def predict_image(img: np.ndarray, query: np.ndarray) -> list:

    #---------------------------------
    MIN_MATCH_COUNT = 10
    MIN_MATCH_COUNT = 10
    matchesMask = 0
    list_of_bboxes = []
    scale = 1
    d_scale = 1
    x = None
    y = None
    dx = 0
    dy = 0
    miss = 0
    d_prv = None
    #-----------------------------------

    img1 = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)         # queryImage
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # trainImage

    if img2.shape[0] > 1400:
        scale = 2
        d_scale = 0.8

    img1_scaled = cv2.resize(img1.copy(), (int(img1.shape[1]/scale), int(img1.shape[0]/scale)))
    img2_scaled = cv2.resize(img2.copy(), (int(img2.shape[1]/scale), int(img2.shape[0]/scale)))
    #----------------------------------------------------------------------------------------------
    while True:
        #delete recognized object
        if x!= None and y!=None :
            img2_scaled[x:x+dx,y:y+dy] = 0

        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1_scaled,None)
        kp2, des2 = sift.detectAndCompute(img2_scaled,None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < d_scale*n.distance:
                good.append(m)
        # print(len(good))
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = img1_scaled.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            x = int(dst[0][0,1])
            y = int(dst[0][0,0])
            min = dst[0]
            max = dst[2]
            d = np.abs(max - min)
            #---------------
            if np.all(d == d_prv):
                break
            d_prv = d
            #---------------
            dx = int(d[0,1])
            dy = int(d[0,0])
            
            print('\n', dx, dy)

            if (dx > 0.4*img1_scaled.shape[0]  and dx < 1.6*img1_scaled.shape[0]) and \
                (dy > 0.4*img1_scaled.shape[1] and dy < 1.6*img1_scaled.shape[1]):
                list_of_bboxes.append((x/img2_scaled.shape[0],y/img2_scaled.shape[1], dx/img2_scaled.shape[0], dy/img2_scaled.shape[1]))
                # list_of_bboxes.append((x/img2_scaled.shape[0],y/img2_scaled.shape[1], 1, 1))
                print('\rObjects found:', len(list_of_bboxes), end="")
                miss = 0
            else:
                miss +=1
                print('pass')
            if miss > 5:
                break
        else:
            print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None
            print('Completed')
            break

    return list_of_bboxes
