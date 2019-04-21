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

# program initialization
try:
    with open(r"tesseract-path.txt", "r") as f:
        pytesseract.pytesseract.tesseract_cmd = f.read()
except:
    print("Error while accessing tesseract path file")
    exit()


# automatic test methods
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


# image processing methods
def apply_threshold(img, thresh=127, maxval=255, show_img=False):
    """
    Applies a threshold to the image and returns it.
    """
    img_thresh = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)[1]

    if show_img:
        show(img_thresh, "After applying threshold")

    return img_thresh

def blur(img, show_img=False):
    """
    Blurs the image and returs it.
    """
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    if show_img:
        show(img_blur, "Blurred image")

    return img_blur

def compute_black_white_ratio(img):
    """
    Computes and returns the black and white pixels ratio for the given image.
    """
    blacks = sum([1 for x in range(img.shape[1]) for y in range(img.shape[0]) if img[y][x] == 0])
    whites = img.shape[0]*img.shape[1] - blacks
    return blacks / whites

def copy_onto_white_square(img, margin=50, show_img=False):
    """
    Copies the given image inside a white square and with a given margin.
    """
    if img.shape[0] > img.shape[1]:
        square_width = img.shape[0] + 2*margin
        y_offset = margin
        x_offset = (img.shape[0] - img.shape[1])//2 + margin
    else:
        square_width = img.shape[1] + 2*margin
        y_offset = (img.shape[1] - img.shape[0])//2 + margin
        x_offset = margin

    white_square = np.zeros((square_width, square_width, 3), np.uint8)
    white_square[:, 0:square_width//2] = (255, 255, 255)
    white_square[:, square_width//2:square_width] = (255, 255, 255)

    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            white_square[y+y_offset][x+x_offset] = img[y][x]

    if show_img:
        show(white_square, "Square image")

    return cv2.cvtColor(white_square, cv2.COLOR_BGR2GRAY)

def dilate(img, kernel=None, show_img=False):
    """
    Dilates given image.
    """
    kernel = np.ones((5, 5), np.uint8) if kernel is None else kernel
    img_dilated = cv2.erode(img, kernel, iterations = 1) # use erode as the image's element is in black and not white!!
    
    if show_img:
        show(img_dilated, "Dilated image")

    return img_dilated

def dilate_until_bw_ratio_reached(img, target_bw_ratio=0.15, kernel=None, show_img=False):
    """
    Dilates given image until desired black and white pixel ratio is reached.
    """
    img_dilated = img.copy()
    kernel = np.ones((7, 7), np.uint8) if kernel is None else kernel

    while compute_black_white_ratio(img_dilated) < target_bw_ratio:
        img_dilated = cv2.erode(img_dilated, kernel, iterations = 1) # use erode as the image's element is in black and not white!!

    if show_img:
        show(img_dilated, "Dilated image to adjust black/white ratio")

    return img_dilated

def erode(img, kernel=None, show_img=False):
    """
    Erodes given image to remove noise.
    """
    kernel = np.ones((5, 5), np.uint8) if kernel is None else kernel
    img_eroded = cv2.dilate(img, kernel, iterations = 1) # use dilate as the image's element is in black and not white!!

    if show_img:
        show(img_eroded, "Eroded image")

    return img_eroded

def extract_elements(img, show_img=False):
    """
    Identifies and returns the different image's elements.
    """
    elements = []

    # search x bounds:
    x_bounds = search_all_elements_x_bounds(img)

    img_y_extremas = [img.shape[1], -1]
    if len(x_bounds) > 0:
        # search y bounds
        for element_index in range(len(x_bounds)//2): # for each element
            element_y_bounds = search_element_y_bounds(img,
                                                       x_bounds[0 + element_index*2],
                                                       x_bounds[1 + element_index*2] + 1,
                                                       img_y_extremas)

            # if bounding rect is defined
            if element_y_bounds[0] is not img_y_extremas[0] and element_y_bounds[1] is not img_y_extremas[1]:
                elements.append(img[element_y_bounds[0]:element_y_bounds[1]+1,
                                    x_bounds[0 + element_index*2]:x_bounds[1 + element_index*2]+1])
                if show_img:
                    show(elements[-1], "Extracted element")

    return elements

def extract_elements_opencv(img, show_img=False):
    """
    Identifies and returns the different image's elements (uses OpenCV findContours() method).
    """
    contours = cv2.findContours(img, 1, 2)[1]
    elements = []
    for cnt in contours[:-1]:
        x, y, w, h = cv2.boundingRect(cnt)
        elements.append(img[y:y+h, x:x+w])
        if show_img:
            show(elements[-1], "Extracted element")

    return elements

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
        if show_img:
            show(img, "Rotated image")
        return img
    if w < h:
        angle = angle + 90

    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1) # angle + 90
    img_rotated = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

    if show_img:
        show(img_rotated, "Rotated image")

    return img_rotated

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

    if show_img:
        box = cv2.minAreaRect(np.array(contours_in_one))
        box = cv2.boxPoints(box)
        box = np.int0(box)
        img2 = img.copy()
        cv2.drawContours(img2, [box] , 0, (0, 0, 255), 2)
        show(img2, "Elements min rect")

    return np.array(contours_in_one)

def load_and_resize(img_path, default_width=600, show_img=False):
    """
    Loads and resizes the image to the specified width. Width/height ratio is not changed.
    Returns the image.
    """
    img = cv2.imread(img_path, 0)
    img = resize(img, default_width)

    if show_img:
        show(img, "Default image")

    return img

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

def search_all_elements_x_bounds(img):
    """
    Searches image's elements x bounds.
    """
    # bounds: first and last columns index containing a black pixel with columns containing at least one black pixel in between
    x_bounds = []
    for x in range(img.shape[1]):
        for y in range(img.shape[0]): # search in each column
            white_column = True
            if img[y][x] == 0: # black pixel
                white_column = False
                if len(x_bounds) % 2 == 0: # found first black pixel of new element
                    x_bounds.append(x) # element's left bound
                break
        if len(x_bounds) % 2 == 1 and white_column: # found first white pixel after element
            x_bounds.append(x-1) # element's right bound
    return x_bounds

def search_element_y_bounds(img, left_x_bound, right_x_bound, img_y_extremas):
    """
    Searches y bounds for element within given x range.
    """
    y_bounds = img_y_extremas.copy()
    for x in range(left_x_bound, right_x_bound): # element's x bounds
        for y in range(img.shape[0]):
            if img[y][x] == 0:
                # search bounds
                if y < y_bounds[0]:
                    y_bounds[0] = y
                if y > y_bounds[1]:
                    y_bounds[1] = y
    return y_bounds

def show(img, window_title):
    """
    Displays given image.
    """
    cv2.imshow(window_title, img)
    cv2.waitKey(0)

def swap_bw(img, show_img=False):
    """
    Modifies and returns the image as black pixels becomes white and whites becomes blacks.
    """
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img[y][x] == 0:
                img[y][x] = 255
            else:
                img[y][x] = 0

    if show_img:
        show(img, "Swapped blacks and whites")

    return img


# prediction methods and global pipeline
def retrieve_equation(elements, model, show_img=False):
    """
    Uses MNIST and Tesseract to predict what represents each image's element and retrieve
    the complete equation.
    Returns the equations as a string.
    """
    equation_text = ""
    for element in elements:
        # predict with MNIST
        prediction = mnist_predict(element, model, show_img=show_img)
        prediction_index = np.argmax(prediction)
        prediction_probability = prediction[prediction_index]

        if prediction_probability <= 1.0:
            # if MNIST prediction's probability is low it's probabily because it's not a digit
            # -> let tesseract do its prediction
            tesseract_prediction = tesseract_predict(element)
            if len(tesseract_prediction) == 1 and tesseract_prediction in ["+", "-", "*", "/"]:
                equation_text += tesseract_prediction
                continue
        equation_text += str(prediction_index)
    return equation_text

def mnist_predict(img, model, show_img=False):
    """
    Uses the given MNSIT model to predict which digit's on the given image.
    Returns a dictionnary with digits as keys and probabilities as values.
    """
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    img = resize_square(img, 28, show_img)
    img = swap_bw(img, show_img)
    img = img.reshape(-1, 28, 28, 1)
    return model.predict(img)[0]

def tesseract_predict(img, config="--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789+-x/"):
    """
    Uses Tesseract to predict which digit's on the given image.
    Returns a string containing the predicted digit.
    """
    return pytesseract.image_to_string(img, lang="eng", config=config)

def process_image(img_path, model, show_img=False):
    """
    Reads image and predicts which equation it contains.
    Returns a string containing concatenated digits and elements (i.e. '2+3').
    """
    # ------------------- load the image -------------------
    img = load_and_resize(img_path, show_img=show_img)

    # ------------------- blur the image -------------------
    img = blur(img, show_img=show_img)

    # ------------------- transform the image in B/W -------------------
    img = apply_threshold(img, show_img=show_img)

    # ------------------- erode the image to remove noise -------------------
    img = erode(img, show_img=show_img)

    # ------------------- dilate the image to restore lost element's weight after erosion -------------------
    img = dilate(img, show_img=show_img)

    # ------------------- find contours of the image -------------------
    contours_in_one = get_all_contours(img, show_img=show_img)

    # ------------------- find the bounding rect of the image and rotate -------------------
    img = fix_rotation(img, contours_in_one, show_img=show_img)

    # ------------------- extract the equation's elements from the image -------------------
    elements = extract_elements(img, show_img=show_img)

    # ------------------- copy each element onto a white square with a margin -------------------
    elements = [copy_onto_white_square(element, show_img=show_img) for element in elements]

    # ------------------- dilate each element to obtain minimal black/white ratio -------------------
    elements = [dilate_until_bw_ratio_reached(element, show_img=show_img) for element in elements]

    # ------------------- mnist/tesseract prediction -------------------
    equation_text = retrieve_equation(elements, model, show_img=show_img)

    if show_img:
        cv2.destroyAllWindows()

    return equation_text


if __name__ == "__main__":
    # create_test_images("additions")
    # test_generated_operations("additions")

    model = load_model("model/mnist_DNN.h5")
    # model.summary()

    equation_text = process_image("img/hw_add_rot.jpg", model, show_img=False)

    print(f"Your equation: {equation_text}")

    # solve_equation(equation_text)
