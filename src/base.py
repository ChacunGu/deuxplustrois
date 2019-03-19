"""
sources :

https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
http://blog.ayoungprogrammer.com/2013/01/equation-ocr-part-1-using-contours-to.html/
"""

import numpy as np
import cv2
from PIL import Image
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"D:\programmes\OCR_Tesseract\Tesseract-OCR\tesseract.exe"


def show(img, window_title):
    cv2.imshow(window_title, img)
    cv2.waitKey(0)

def getAllContours(img):
    im, contours, hierarchy = cv2.findContours(img, 1, 2)
    contours_in_one = []
    for cnt in contours[:-1]:
            for point in cnt:
                contours_in_one.append(point[0].tolist())
    return np.array(contours_in_one)

def fixRotation(img, contours_in_one):
    box = cv2.minAreaRect(contours_in_one) # returns a box2D object ( center (x,y), (width, height), angle of rotation ).
    w = box[1][0]
    h = box[1][1]
    x = box[0][0] - h/2
    y = box[0][1] - w/2
    angle = box[2]
    w, h, x, y = int(w), int(h), int(x), int(y)

    # box = cv2.boxPoints(box)
    # box = np.int0(box)
    # img2 = img
    # cv2.drawContours(img2,[box],0,(0,0,255),2)
    # show(img2, "image2")

    if w < h:
        angle = angle + 90

    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1) # angle + 90

    return cv2.warpAffine(img,M,(cols,rows))



if __name__ == "__main__":
    # ------------------- load the image -------------------
    # img = cv2.imread('..\img\\base.jpg',0)
    img = cv2.imread('..\img\\rotated_basic_addition.jpg',0)
    show(img, "image")


    # ------------------- blur the image -------------------
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    show(img_blur, "image blurred")

    # ------------------- transform the image in B/W -------------------
    ret, img_thresh = cv2.threshold(img_blur,127,255,cv2.THRESH_BINARY)
    show(img_thresh, "image treshold")

    # ------------------- find contours of the image -------------------
    contours_in_one = getAllContours(img)

    # ------------------- find the bounding rect of the image and rotate -------------------
    img_rotated = fixRotation(img ,contours_in_one)
    show(img_rotated, "image rotated")

    cv2.destroyAllWindows()






    # img_bounding_rect2 = img
    # cv2.rectangle(img_bounding_rect2,(x, y),(x+h, y+w),(0,255,0),2)
    # rect = cv2.boxPoints(box) # get the points of the box
    # rect = np.int0(rect)
    # cv2.drawContours(img_bounding_rect2,[rect],0,(0,0,255),2)
    # cv2.imshow('image bounding rect 2', img_bounding_rect2)
    # cv2.waitKey(0)

    # crop_img = img[y:y+w, x:x+h]
    # cv2.imshow('image crop', crop_img)
    # cv2.waitKey(0)

    # ------------------- extract the contours -------------------
    # im, contours, hierarchy = cv2.findContours(img_thresh, 1, 2)
    # image_individual_bounding_rect = img
    # elements = []
    # for cnt in contours[:]:
    #         x,y,w,h = cv2.boundingRect(cnt)
    #         bounding_rect = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
    #         cv2.rectangle(image_individual_bounding_rect,tuple(bounding_rect[0]),tuple(bounding_rect[2]),(0,255,0),2)
    #         elements.append(img[y:y+h, x:x+w])
    #
    # cv2.imshow('image individual bounding rect', image_individual_bounding_rect)
    # cv2.waitKey(0)
    #
    # i = 0
    # for elem in elements:
    #     cv2.imshow('elem%d' % i, elem)
    #     cv2.waitKey(0)
    #     i += 1
