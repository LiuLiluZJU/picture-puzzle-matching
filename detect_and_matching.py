import os
import numpy as np
import cv2
from timeit import default_timer as timer



def GetROI(image, background, show_image_flag=0):
    """返回图像中感兴趣物体的包围框和掩模
    
    @param image：输入图像
    @param background：图像背景
    @param show_image_flag：是否显示中间结果，默认为不显示（0）
    @return bboxes：感兴趣物体的矩形包围框
    @return mask：感兴趣物体的掩模

    """

    # 完整图减去背景
    backsub_res = cv2.subtract(image, background)

    # 灰度化
    gray_img = cv2.cvtColor(backsub_res, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    retval,threshold = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)

    # 中值滤波
    threshold = cv2.medianBlur(threshold, 3)

    # 先膨胀后腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    
    # 先腐蚀后膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 寻找轮廓（掩模形式）
    mask = np.zeros_like(image)
    _, contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)
    mask[mask > 0] = 1

    # 寻找轮廓的包围框（轮廓包围面积大于1000的被选择）
    image_show = image.copy()
    bboxes = []
    for c in contours:
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append((x, y, w, h))
            cv2.rectangle(image_show, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
    # 显示
    if show_image_flag:
        cv2.imshow('backsub_res', backsub_res)
        cv2.waitKey(0)
        cv2.imshow('gray_img', gray_img)
        cv2.waitKey(0)
        cv2.imshow('threshold', threshold)
        cv2.waitKey(0)
        cv2.imshow('closed', closed)
        cv2.waitKey(0)
        cv2.imshow('opened', opened)
        cv2.waitKey(0)
        cv2.imshow('scene_img', image_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return bboxes, mask



if __name__ == "__main__":

    tic = timer()

    # 读取图像
    background = cv2.imread("./images2/background.jpeg")
    full_jigsaw = cv2.imread("./images2/full.jpeg")
    scene_img = cv2.imread("./images2/scene_1.jpeg")

    # 提取原图中的ROI(包围框、掩模)及目标图
    ROIs_target, mask_target = GetROI(full_jigsaw, background, 0)
    box_target = ROIs_target[0]
    full_jigsaw = np.multiply(full_jigsaw, mask_target)  # 使用掩模
    cv2.imshow('full_jigsaw', full_jigsaw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 提取散乱图中的ROI(包围框、掩模)
    ROIs_template, mask_template = GetROI(scene_img, background, 1)
    scene_img = np.multiply(scene_img, mask_template)  # 使用掩模

    # 初始化SIFT提取器
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=1, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    # sift = cv2.xfeatures2d.SIFT_create()

    # 切分目标图，一共num_x * num_y块子图，相邻子图之间有一半重叠
    num_x = 4
    num_y = 4
    for i in range(num_x):
        for j in range(num_y):

            # 利用原图ROI提取目标图
            interval_x = int(box_target[2] / (num_x + 1))
            interval_y = int(box_target[3] / (num_y + 1))
            start_x = box_target[0] + interval_x * i
            start_y = box_target[1] + interval_y * j
            stop_x = start_x + 2 * interval_x
            stop_y = start_y + 2 * interval_y
            target = full_jigsaw[start_y : stop_y, start_x : stop_x]

            # 使用SIFT提取关键点和描述子
            kp2, des2 = sift.detectAndCompute(target, None)
            target_img = cv2.drawKeypoints(target, kp2, None)

            for box_template in ROIs_template:
                # 利用散乱图的ROI裁剪模板图
                template = scene_img[box_template[1] : box_template[1] + box_template[3], box_template[0] : box_template[0] + box_template[2]]

                # 使用SIFT提取关键点和描述子
                kp1, des1 = sift.detectAndCompute(template,None)
                template_img = cv2.drawKeypoints(template, kp1, None)
                # cv2.imshow('template_img', template_img)
                # cv2.waitKey(0)

                # 关键点匹配
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                # flann = cv2.FlannBasedMatcher(index_params, search_params)
                flann = cv2.BFMatcher()
                matches = flann.knnMatch(des1, des2, k=2)

                # 保存匹配良好的点对
                good = []
                for m, n in matches:
                    if m.distance < 0.95 * n.distance:
                        good.append(m)

                # 计算变换矩阵
                MIN_MATCH_COUNT = 10
                if len(good) > MIN_MATCH_COUNT:
                    # 获取关键点坐标
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    # 保存匹配点对坐标
                    # src_pts_global = src_pts + np.array([box_template[0], box_template[1]])
                    # dst_pts_global = dst_pts + np.array([box_target[0], box_target[1]])
                    # with open("./matching.txt", 'w') as f:
                    #     f.write('map index + query index \r\n')
                    #     for k in range(dst_pts_global.shape[0]):
                    #         line1 = np.squeeze(dst_pts_global[k])
                    #         line2 = np.squeeze(src_pts_global[k])
                    #         f.write(str(line1[0]) + ' ' + str(line1[1]) + ' ' + str(line2[0]) + ' ' + str(line2[1]) + '\r\n')
                    
                    # 计算单应矩阵
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    matchesMask = mask.ravel().tolist()
                    h,w,c = template.shape

                    # 变换模板图
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts,M)
                    # cv2.polylines(target,[np.int32(dst)],True,0,2, cv2.LINE_AA)  # 画出变换后的外包围框
                else:
                    print( "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
                    matchesMask = None

                draw_params = dict(matchColor=(0, 255, 0), 
                                singlePointColor=None,
                                matchesMask=matchesMask, 
                                flags=2)
                result = cv2.drawMatches(template,kp1,target,kp2,good,None,**draw_params)
                cv2.imshow('result', result)
                cv2.waitKey(0)
                cv2.imwrite("./{}.jpg".format(num_y * i + j), result)

            cv2.destroyAllWindows()

    toc = timer() 
    print('template match using time {}'.format(toc - tic))  
            

