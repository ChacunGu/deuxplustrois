"""
sources :

https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
http://blog.ayoungprogrammer.com/2013/01/equation-ocr-part-1-using-contours-to.html/
"""

import cv2
from keras.models import load_model
import numpy as np
import os
from PIL import Image
import pytesseract
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

try:
    with open(r"tesseract-path.txt", "r") as f:
        pytesseract.pytesseract.tesseract_cmd = f.read()
except:
    print("Error while accessing tesseract path file")
    exit()


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
            output = process_image(img_path, False)
            success = input==output
            print(input, output, success)
            total_success += 1 if success else 0
    print("Success percentage:", (total_success/(limit_left_op*limit_right_op))*100, "%")

def show(img, window_title):
    """
    Displays given image.
    """
    cv2.imshow(window_title, img)
    cv2.waitKey(0)

def get_all_contours(img, show_img=False):
    """
    Returns an array containing each point forming a contour of any element in the image.
    """
    contours = cv2.findContours(img, 1, 2)[1]
    contours_in_one = []
    for cnt in contours[:-1]:
        for point in cnt:
            contours_in_one.append(point[0].tolist())

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if show_img:
            img2 = img.copy()
            cv2.drawContours(img2, [box] , 0, (0, 0, 255), 2)
            show(img2, "Image element's contour")

    return np.array(contours_in_one)

def fix_rotation(img, contours_in_one, show_img=False):
    """
    Rotates the image until it's perfectly horizontal.
    """
    box = cv2.minAreaRect(contours_in_one) # returns a box2D object ( center (x,y), (width, height), angle of rotation ).
    w = box[1][0]
    h = box[1][1]
    x = box[0][0] - h/2
    y = box[0][1] - w/2
    angle = box[2]
    w, h, x, y = int(w), int(h), int(x), int(y)

    if angle > -5 or angle < -85:
        return img
    if w < h:
        angle = angle + 90

    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1) # angle + 90
    img_rotated = cv2.warpAffine(img, M, (cols, rows))

    if show_img:
        show(img_rotated, "Rotated image")

    return img_rotated

def extract_elements(img, show_img=False):
    """
    Identifies and returns the different image's elements.
    """
    contours = cv2.findContours(img, 1, 2)[1]
    elements = []
    for cnt in contours[:-1]:
        x, y, w, h = cv2.boundingRect(cnt)
        elements.append(img[y:y+h, x:x+w])
        if show_img:
            show(elements[-1], "Extracted element")

    return elements

def resize(img, width, show_img=False):
    """
    Resizes the given image with the given ratio (old_width/width).
    """
    ratio = img.shape[1] / width
    width = int(img.shape[1] / ratio)
    height = int(img.shape[0] / ratio)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    if show_img:
        show(resized, "Resized image")

    return resized

def resize_square(img, width, show_img=False):
    """
    Resizes the image to square dimensions for given width.
    """
    dim = (width, width)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    if show_img:
        show(resized, "Resized image")

    return resized

def swap_bw(img, show_img=False):
    """
    Modifies and returns the image as black pixels becomes white and whites becomes blacks.
    """
    for x in range(len(img)):
        for y in range(len(img[x])):
            if img[x][y] == 0:
                img[x][y] = 255
            else:
                img[x][y] = 0
    
    if show_img:
        show(img, "Swapped blacks and whites")

    return img

def mnist_predict(img, model, show_img=False):
    """
    Uses the given MNSIT model to predict which digit's on the given image. 
    Returns a dictionnary with digits as keys and probabilities as values.
    """
    img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]
    img = resize_square(img, 28, show_img)
    img = swap_bw(img, show_img)
    img = img.reshape(-1, 28, 28, 1)
    return model.predict(img)[0] # model.predict_classes(img)

def process_image(img_path, model, show_img=False):
    """
    Reads image and predicts which equation it contains.
    Returns a string containing concatenated digits and elements (i.e. '2+3').
    """
    # ------------------- load the image -------------------
    img = cv2.imread(img_path, 0)
    img = resize(img, 600)
    if show_img:
        show(img, "Default image")

    # ------------------- blur the image -------------------
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    if show_img:
        show(img_blur, "Blurred image")

    # ------------------- transform the image in B/W -------------------
    img_thresh = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)[1]
    if show_img:
        show(img_thresh, "After applying threshold")

    # ------------------- find contours of the image -------------------
    contours_in_one = get_all_contours(img_thresh, show_img=show_img)

    # ------------------- find the bounding rect of the image and rotate -------------------
    img_rotated = fix_rotation(img_thresh, contours_in_one, show_img)

    # ------------------- extract the equation's elements from the image -------------------
    elements = extract_elements(img_rotated, show_img)

    # ------------------- mnist/tesseract prediction -------------------
    equation_text = ""
    for element in elements:
        # predict with MNIST
        prediction = mnist_predict(element, model)
        prediction_index = np.argmax(prediction)
        prediction_probability = prediction[prediction_index]

        if prediction_probability < 1.0: 
            # if MNIST prediction's probability is low it's probabily because it's not a digit
            # -> let tesseract do its prediction
            tesseract_prediction = pytesseract.image_to_string(element, lang="eng", config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789")
            if len(tesseract_prediction) == 1 and tesseract_prediction in ["+", "-", "*", "/"]:
                equation_text += tesseract_prediction
                continue
        equation_text += str(prediction_index)

    if show_img:
        cv2.destroyAllWindows()

    return equation_text


if __name__ == "__main__":
    # create_test_images("additions")
    # test_generated_operations("additions")

    model = load_model("model/mnist_DNN.h5")
    # model.summary()

    equation_text = process_image("img/clean_basic_addition.jpg", model, show_img=True)

    print(equation_text)

    # solve_equation(equation_text)
