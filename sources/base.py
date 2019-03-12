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

def create_test_images(sub_directory, limit_left_op=10, limit_right_op=10, operator="+", font=cv2.FONT_HERSHEY_DUPLEX):
    """
    Creates test images containg each numbers between 0 and parameter limit_left_op an operator and another
    number between 0 and parameter limit_right_op.
    """
    EMPTY_IMAGE = r'../img/clean_empty.jpg'
    DESTINATION_DIRECTORY = f'../img/generated/{sub_directory}/'
    
    for i in range(limit_left_op):
        for j in range(limit_left_op):
            text = f'{i}{operator}{j}' # define text
            img = cv2.imread(EMPTY_IMAGE, 1) # load empty image
            cv2.putText(img, text, (10,100), font, 1, (0, 0, 0), 2, cv2.LINE_AA) # write text on empty image
            cv2.imwrite(DESTINATION_DIRECTORY + f'{i}_{j}.jpg', img) # save new image

def test_generated_operations(sub_directory, limit_left_op=10, limit_right_op=10, operator="+"):
    """
    Loads generated images from a subdirectory recreates input and compares with expected output.
    """
    SOURCE_DIRECTORY = f'../img/generated/{sub_directory}'
    total_success = 0
    for i in range(limit_left_op):
        for j in range(limit_right_op):
            img_path = f"{SOURCE_DIRECTORY}/{i}_{j}.jpg"
            input = f'{i}{operator}{j}'
            output = process_image(img_path)
            success = input==output
            print(input, output, success)
            total_success += 1 if success else 0
    print("Success percentage:", (total_success/(limit_left_op*limit_right_op))*100, "%")

try:
    with open(r"..\tessereact-path.txt", "r") as f:
        pytesseract.pytesseract.tesseract_cmd = f.read()
except:
    print("Error while accessing tessereact path file")
    exit()

def process_image(img_path):
    # ------------------- load the image -------------------
    img = cv2.imread(img_path,0)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)

    # ------------------- blur the image -------------------
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    # cv2.imshow('image blur', img_blur)
    # cv2.waitKey(0)

    # ------------------- transform the image in B/W -------------------
    ret, img_thresh = cv2.threshold(img_blur,127,255,cv2.THRESH_BINARY)
    # cv2.imshow('image threshold', img_thresh)
    # cv2.waitKey(0)

    # ------------------- find contours of the image -------------------
    im, contours, hierarchy = cv2.findContours(img_thresh, 1, 2)
    contours_in_one = []
    for cnt in contours[:-1]:
        for point in cnt:
            contours_in_one.append(point[0].tolist())
    contours_in_one = np.array(contours_in_one)

    # ------------------- find the bounding rect of the image and rotate -------------------
    # box = cv2.minAreaRect(contours_in_one) # returns a box2D object ( center (x,y), (width, height), angle of rotation ).
    # w = box[1][0]
    # h = box[1][1]
    # x = box[0][0] - h/2
    # y = box[0][1] - w/2
    # angle = box[2]
    # w, h, x, y = int(w), int(h), int(x), int(y)

    # print(-angle)
    # rows,cols = img.shape
    # M = cv2.getRotationMatrix2D((cols/2,rows/2),angle+90,1) # angle + 90
    # img = cv2.warpAffine(img,M,(cols,rows))
    # cv2.imshow('image test', img)
    # cv2.waitKey(0)

    img_test = img.copy()

    # no comment _-----_
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

    #------------------- blur the image -------------------
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    # cv2.imshow('rotated image blur', img_blur)
    # cv2.waitKey(0)

    #------------------- transform the image in B/W -------------------
    ret, img_thresh = cv2.threshold(img_blur,127,255,cv2.THRESH_BINARY)
    # cv2.imshow('rotated image threshold', img_thresh)
    # cv2.waitKey(0)

    # ------------------- extract the contours -------------------
    im, contours, hierarchy = cv2.findContours(img_thresh, 1, 2)
    image_individual_bounding_rect = img
    elements = []
    for cnt in contours[:-1]:
        x,y,w,h = cv2.boundingRect(cnt)
        bounding_rect = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
        cv2.rectangle(image_individual_bounding_rect,tuple(bounding_rect[0]),tuple(bounding_rect[2]),(0,255,0),2)
        elements.append(img_test[y:y+h, x:x+w])

    # cv2.imshow('image individual bounding rect', image_individual_bounding_rect)
    # cv2.waitKey(0)

    # i = 0
    # for elem in elements:
    #     cv2.imshow('elem%d' % i, elem)
    #     cv2.waitKey(0)
    #     i += 1

    # ------------------- tesseract part -------------------

    text = pytesseract.image_to_string(img_thresh, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    # print("full image", text)
    return text

    # for elem in elements:
    #     cv2.imshow('elem : %s' % (text), elem)
    #     # text = pytesseract.image_to_string(elem)
    #     # text = pytesseract.image_to_string(elem, lang='eng', config='digits')
    #     text = pytesseract.image_to_string(elem, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    #     print(text)
    #     cv2.waitKey(0)


    cv2.destroyAllWindows()




if __name__ == "__main__":
    # create_test_images("additions")
    test_generated_operations("additions")