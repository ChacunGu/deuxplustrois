"""
sources :

https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
http://blog.ayoungprogrammer.com/2013/01/equation-ocr-part-1-using-contours-to.html/
"""

import copy
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
    Images name must be of format 'i_j.jpg' with i and j starting at 0 and lineary incremented up to
    parameters 'limit_left_op' and 'limit_right_op'. The predicted values are those i + j.
    I.e. '0_0.jpg', '0_1.jpg', ...
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
def adjust_element_weight(img, target_bw_min_ratio=0.15, target_bw_max_ratio=0.20, kernel=None, show_img=False):
    """
    Adjusts element's weight by dilating or eroding until the targeted ratio is reached.
    """
    kernel = np.ones((7, 7), np.uint8) if kernel is None else kernel

    if compute_black_white_ratio(img) < target_bw_min_ratio:
        img = dilate_until_bw_ratio_reached(img, kernel, target_bw_min_ratio)
    elif compute_black_white_ratio(img) > target_bw_max_ratio:
        img = erode_until_bw_ratio_reached(img, kernel, target_bw_max_ratio)

    if show_img:
        show(img, "Transformed image to adjust black/white ratio")

    return img

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

def copy_onto_white_square(img, img_size=200, show_img=False):
    """
    Copies the given image inside a white square and with a given margin.
    """
    margin = int(img_size/100 * 20)
    if img.shape[0] > img.shape[1]:
        img = resize(img, img_size, False, show_img=show_img)
        square_width = img.shape[0] + 2*margin
        y_offset = margin
        x_offset = (img.shape[0] - img.shape[1])//2 + margin
    else:
        img = resize(img, img_size, True, show_img=show_img)
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

def dilate_until_bw_ratio_reached(img, kernel, target_bw_min_ratio=0.15):
    """
    Dilates given image until desired black and white pixel ratio is reached.
    """
    img_dilated = img.copy()
    while compute_black_white_ratio(img_dilated) < target_bw_min_ratio:
        img_dilated = cv2.erode(img_dilated, kernel, iterations = 1) # use erode as the image's element is in black and not white!!

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

def erode_until_bw_ratio_reached(img, kernel, target_bw_max_ratio=0.40):
    """
    Erodes given image until desired black and white pixel ratio is reached.
    """
    img_eroded = img.copy()
    while compute_black_white_ratio(img_eroded) > target_bw_max_ratio:
        img_eroded = cv2.dilate(img_eroded, kernel, iterations = 1) # use dilate as the image's element is in black and not white!!

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
    if img is None:
        print(f"File has not been found or is of an unsupported format: {img_path}")
        exit()

    img = resize(img, default_width, False)

    if show_img:
        show(img, "Default image")

    return img

def resize(img, size, resize_by_width=True, show_img=False):
    """
    Resizes the given image with the given ratio (old_size/size) by width or height.
    """
    ratio = (img.shape[1] if resize_by_width else img.shape[0]) / size
        
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
def extract_elements_from_string_equation(equation):
    """
    Reads given equation as string, converts it to digits and signs.
    Returns an array containing in each element a subarray with at index 0 the extracted element's
    type ("digit" or "sign") and at index 1 the element (int for a digit and string for a sign). Returns
    also the number of digits and signs found.
    I.e. [["digit", 2], ["sign", "+"], ["digit", 3]], 2, 1
    """
    elements = []
    nb_digits = 0
    nb_signs = 0

    # retrieve equation's elements
    for c in equation:
        try:
            digit = str(int(c))
            if len(elements) > 0:
                if elements[-1][0] == "digit":
                    elements[-1][1] += digit # multiple digits number
                    continue
            elements.append(["digit", digit]) # new number
            nb_digits += 1
        except:
            sign = c
            if len(elements) <= 0 or elements[-1][0] == "sign" or sign == "=":
                continue # invalid
            elements.append(["sign", sign]) # new sign
            nb_signs += 1
    return elements, nb_digits, nb_signs

def compute_equation_result(elements):
    """
    Uses reduction to compute and finally return the equation's result.
    """
    elements = copy.deepcopy(elements)
    for sign in [("x", "/"), ("+", "-")]: # respect operations priority
        for i in range(0, len(elements)-1, 2):
            if elements[i+1][1] in sign:
                if elements[i+1][1] == "x":
                    elements[i+2][1] = float(elements[i+0][1]) * float(elements[i+2][1])
                elif elements[i+1][1] == "/":
                    elements[i+2][1] = float(elements[i+0][1]) / float(elements[i+2][1])
                elif elements[i+1][1] == "+":
                    elements[i+2][1] = float(elements[i+0][1]) + float(elements[i+2][1])
                elif elements[i+1][1] == "-":
                    elements[i+2][1] = float(elements[i+0][1]) - float(elements[i+2][1])
                elements[i+0] = None
                elements[i+1] = None
        
        # remove None elements
        elements = [element for element in elements if element is not None]
    
    if (len(elements) > 1):
        return None    
    return elements[0][1]

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

def process_image(img_path, model, favor_tesseract=True, show_img=False):
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

    # ------------------- dilate or erode each element to obtain minimal black/white ratio -------------------
    elements = [adjust_element_weight(element, show_img=show_img) for element in elements]

    # ------------------- mnist/tesseract prediction -------------------
    equation_text = retrieve_equation(elements, model, favor_tesseract, show_img=show_img)

    if show_img:
        cv2.destroyAllWindows()

    return equation_text

def retrieve_equation(elements, model, favor_tesseract=True, show_img=False):
    """
    Uses MNIST and Tesseract to predict what represents each image's element and retrieve
    the complete equation.
    Returns the equations as a string.
    """
    equation_text = ""
    for element in elements:
        # predict with MNIST
        mnist_prediction = mnist_predict(element, model, show_img=show_img)
        mnist_prediction = str(np.argmax(mnist_prediction))

        # predict with Tesseract
        tesseract_prediction = tesseract_predict(element)

        # choose a prediction
        print(mnist_prediction, tesseract_prediction)
        if tesseract_prediction == "" and mnist_prediction in ["2", "5"]: # detect '-' like a brute as Tesseract can't handle it
            equation_text += "-"
        else:
            if favor_tesseract:
                equation_text += mnist_prediction if tesseract_prediction == "" else tesseract_prediction
            else:
                equation_text += tesseract_prediction if tesseract_prediction in ["+", "-", "x", "/", "="] else mnist_prediction
    return equation_text

def solve_equation(equation):
    """
    Reads given equation as string, converts it to digits and signs, computes the result and displays it.
    """
    elements, nb_digits, nb_signs = extract_elements_from_string_equation(equation)
    
    # compute equation's result if structure is valid
    if len(elements) > 0 and elements[0][0] == "digit" and elements[-1][0] == "digit" and nb_digits == nb_signs+1 and nb_digits >= 2:
        result = compute_equation_result(elements)
        if result is not None:
            solved_equation = " ".join([element[1] for element in elements])
            solved_equation += f" = {result}"
            return solved_equation
    print(f"Unable to solve this equation: {equation}")
    exit()

def tesseract_predict(img, config="--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789+x/-="):
    """
    Uses Tesseract to predict which digit's on the given image.
    Returns a string containing the predicted digit.
    """
    return pytesseract.image_to_string(img, lang="eng", config=config)


if __name__ == "__main__":
    # create_test_images("additions")
    # test_generated_operations("additions")


    model = load_model("model/mnist_DNN.h5")
    # model.summary()

    # equation_text = process_image("img/generated/additions/4_6.jpg", model, favor_tesseract=True, show_img=False)
    # equation_text = process_image("img/hw_add_rot.jpg", model, favor_tesseract=False, show_img=False)
    # equation_text = process_image("img/hw/7mul7.jpg", model, favor_tesseract=False, show_img=True)
    equation_text = process_image("img/hw/complex4.jpg", model, favor_tesseract=False, show_img=True)
    
    solved_equation = solve_equation(equation_text)
    print("Your equation:", solved_equation)
