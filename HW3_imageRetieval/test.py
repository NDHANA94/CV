from IPython.display import clear_output
import cv2
from matplotlib import pyplot as plt
import numpy as np

temp1 = cv2.imread('template_0_0.jpg')
temp2 = cv2.imread('template_0_1.jpg')
temp3 = cv2.imread('template_1.jpg')
temp4 = cv2.imread('template_2.jpg')
temp5 = cv2.imread('template_3.jpg')
temp6 = cv2.imread('template_extreme.jpg')
temp = [temp1, temp2, temp3, temp4, temp5, temp6]

train_0 = cv2.imread('train_0.jpg')
train_1 = cv2.imread('train_1.jpg')
train_2 = cv2.imread('train_2.jpg')
train_3 = cv2.imread('train_3.jpg')
train_4 = cv2.imread('train_extreme.jpg')
train = [train_0, train_1, train_2, train_3, train_4] 

temp_gray = [cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in temp]
temp_rgb = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in temp]
train_gray = [cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in train]
train_rgb = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in train]

MIN_MATCH_COUNT = 10
matchesMask = 0
list_of_bboxes = []
scale = 1
d_scale = 1

img1 = temp_gray[2].copy()          # queryImage
img2 = train_gray[1].copy()# trainImage

if img2.shape[0] > 1400:
    scale = 2
    d_scale = 0.8

img1_resized = cv2.resize(img1.copy(), (int(img1.shape[1]/scale), int(img1.shape[0]/scale)))
img2_resized = cv2.resize(img2.copy(), (int(img2.shape[1]/scale), int(img2.shape[0]/scale)))


delete = True
x = None
y = None
dx = 0
dy = 0
img2_resized_ = img2_resized.copy()

miss = 0
while True:

    if x!= None and y!= None and delete:
        img2_resized_[x:x+dx,y:y+dy] = 0

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1_resized,None)
    kp2, des2 = sift.detectAndCompute(img2_resized_,None)
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
    print(len(good))
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1_resized.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        x = int(dst[0][0,1])
        y = int(dst[0][0,0])
        min = dst[0]
        max = dst[2]
        d = max - min
        dx = int(d[0,1])
        dy = int(d[0,0])
        area = dx*dy
        clear_output()
        # print(d)
        # if dx < img1_resized.shape[0]/2:
        #     delete = False
        #     d_scale += 1
        # else:
        #     delete = True

        if area > 0*img1_resized.shape[0]*img1_resized.shape[1] and area < 1.85*img1_resized.shape[0]*img1_resized.shape[1] :
            list_of_bboxes.append((x/img2_resized_.shape[0],y/img2_resized_.shape[1], dx/img2_resized_.shape[0], dy/img2_resized_.shape[1]))
            print('Objects found:', len(list_of_bboxes))
            miss = 0
        else:
            miss +=1
            print('pass')
        if miss > 5:
            break

        # ========================================================================================
        img2_resized = cv2.polylines(img2_resized_,[np.int32(dst)],True,(255,0,0),5, cv2.LINE_AA)
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2_resized_,kp2,good,None,**draw_params)
        plt.figure(dpi=200)
        plt.imshow(img3, 'gray'),plt.show()
        plt.show()
        # ========================================================================================

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
        print('Completed')
        break

# plt.figure(dpi=150)
# plt.imshow(img2_resized_)
# plt.show()

xMin_list = [n[0]*img2.shape[0] for n in list_of_bboxes]
yMin_list = [n[1]*img2.shape[1] for n in list_of_bboxes]
dx_list = [n[2]*img2.shape[0] for n in list_of_bboxes]
dy_list = [n[3]*img2.shape[1] for n in list_of_bboxes]

plt.figure(dpi=350)
for i in range ( len(xMin_list)):
    plt.plot([yMin_list[i], yMin_list[i] + dy_list[i]], [xMin_list[i], xMin_list[i]], '-r', markersize=0.51)
    plt.plot([yMin_list[i], yMin_list[i]],              [xMin_list[i], xMin_list[i] + dx_list[i]], '-r', markersize=0.51)

    plt.plot([yMin_list[i] + dy_list[i], yMin_list[i]+ dy_list[i]], [xMin_list[i], xMin_list[i]+ dx_list[i]], '-r', markersize=0.51)
    plt.plot([yMin_list[i], yMin_list[i]+ dy_list[i]], [xMin_list[i]+ dx_list[i], xMin_list[i]+ dx_list[i]], '-r', markersize=0.51)
    # plt.plot([100,0], [100,1000], '-r')
plt.imshow(img2)
plt.show()

print(len(list_of_bboxes))
