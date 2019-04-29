"""
deuxplustrois - Image Processing course,
Chacun Guillaume, Graells Noah, Vorpe Fabien
HE-Arc, Neuchâtel, Switzerland
2019

***

deuxplustrois.py
@author: @chacungu, @noahgraells, @fabienvorpe 
"""

import cv2
from deuxplustrois import process_image, solve_equation, create_test_images, test_generated_operations
import os

class Interface:
    def __init__(self, model):
        self.state_id = 1
        self.model = model
        self.filename = None
        self.favor_tesseract = False
        self.show_algorithm_steps = False

    def read_int(self, allowed_range):
        """
        Reads an integer from the given list of the command-line.
        """
        try:
            user_input = int(input())
            while user_input not in allowed_range:
                print(f"Please enter a number from this range: {allowed_range}")
                user_input = int(input())
            return user_input
        except:
            print(f"Please enter a number from this range: {allowed_range}")
            return self.read_int(allowed_range)

    def does_image_exist(self, filename):
        """
        Verifies the given image does exist.
        """
        allowed_file_formats_and_labels = {
            "Always supported: ": ["bmp", "dib", "pbm", "pgm", "ppm", "sr", "ras"],
            "Partially supported: ": ["jpeg", "jpg", "jpe", "jp2", "png", "webp", "sr", "ras", "tiff", "tif"],
            }
        allowed_file_formats = [format for format_list in list(allowed_file_formats_and_labels.values()) for format in format_list]

        try:
            file_extension = filename.split(".")[-1]
            if len(filename.split(".")) > 1 and file_extension in allowed_file_formats:
                img = open(filename)
                if img is not None:
                    return True
            else:
                print("Your image's format is not supported by OpenCV.")
                print("OpenCV only supports those formats:")
                for support, formats  in allowed_file_formats_and_labels.items():
                    print("  ", support, formats)
                return False
        except:
            pass
        print(f"Your image has not been found: {filename}")
        return False

    def read_image_path(self):
        """
        Reads an image path of the command-line.
        """
        try:
            user_input = str(input())
            while not self.does_image_exist(user_input):
                print("")
                print("Please enter your image's path (relative path)")
                user_input = str(input())
            return user_input
        except:
            return None

    def state_1(self):
        """
        Menu n°1: select an operation.
        """
        print("")
        print("*** Welcome on deuxplustrois! ***")
        print("What do you want to do ?")
        print("(1) Load an image and compute equation's result")
        print("(2) Automatic testing")
        print("(3) Exit")

        user_input = self.read_int([1,2,3])

        self.favor_tesseract = False
        self.show_algorithm_steps = False
        
        if user_input == 1:
            self.next_state(3)
        elif user_input == 2:
            self.next_state(8)
        elif user_input == 3:
            print("Goodbye!")
            exit()

    def state_2(self, args):
        """
        Menu n°2: handle command-line arguments.
        """
        # python deuxplustrois.py file
        # python deuxplustrois.py file -show_steps False
        # python deuxplustrois.py file -favor_tesseract True
        # python deuxplustrois.py file -favor_tesseract True -show_steps False
        # python deuxplustrois.py file -favor_tesseract False -show_steps True

        print("")

        args.pop(0)
        if len(args) > 0:
            filename = args.pop(0)
            if self.does_image_exist(filename):
                self.filename = filename

                try:
                    for i in range(0, len(args), 2):
                        if args[i] == "-favor_tesseract":
                            self.favor_tesseract = bool(args[i+1])
                        elif args[i] == "-show_steps":
                            self.show_algorithm_steps = bool(args[i+1])
                        else:
                            print(f"Invalid argument {args[i]}. Argument ignored.")
                except:
                    print("Invalid argument(s)")
                    self.next_state(1)
                    return
                self.next_state(5)
                return
            else:
                self.next_state(1)
                return

        print("Insufficient argument(s)")
        self.next_state(1)
    
    def state_3(self):
        """
        Menu n°3: enter image path.
        """
        print("")
        print("*** Please enter your image's path (relative path) ***")
        
        user_input = self.read_image_path()
        if user_input is not None:
            self.filename = user_input
            self.next_state(4)
        else:
            self.next_state(1)
    
    def state_4(self):
        """
        Menu n°4: display steps ?
        """
        print("")
        print("*** Do you want to see intermediary images (algorithm's steps) ? ***")
        print("(1) Yes")
        print("(2) No")
        print("(3) Back to main menu")
        
        user_input = self.read_int([1,2,3])
        if user_input == 1:
            self.show_algorithm_steps = True
            self.next_state(5)
        elif user_input == 2:
            self.show_algorithm_steps = False
            self.next_state(5)
        elif user_input == 3:
            self.next_state(1)
        
    def state_5(self):
        """
        Menu n°5: program execution
        """
        print("")
        print(f"Processing {self.filename}. Please wait...")
        print("Parameters:")
        print(f"  favor Tesseract over MNIST: {self.favor_tesseract}")
        print(f"  show algorithm steps: {self.show_algorithm_steps}")
        print("")
        equation_text = process_image(self.filename, self.model, favor_tesseract=self.favor_tesseract, show_img=self.show_algorithm_steps)
        solved_equation = solve_equation(equation_text)
        if solved_equation is not None:
            print("Your equation:", solved_equation)
            self.next_state(6)
        else:
            self.next_state(7)
        
    def state_6(self):
        """
        Menu n°6: did we read the equation successfuly ?
        """
        print("")
        print("*** Did we read your equation successfuly ? ***")
        print("(1) Yes")
        print("(2) No")
        print("(3) Back to main menu")
        
        user_input = self.read_int([1,2,3])
        if user_input == 1:
            print("Yeah!")
            self.next_state(1)
        elif user_input == 2:
            self.next_state(7)
        elif user_input == 3:
            self.next_state(1)
        
    def state_7(self):
        """
        Menu n°7: do you want to try another algorithm ?
        """
        print("")
        print("*** Do you want to try with another algorithm ? ***")
        print("(1) Yes")
        print("(2) No")
        print("(3) Back to main menu")
        
        user_input = self.read_int([1,2,3])
        if user_input == 1:
            self.favor_tesseract = not self.favor_tesseract
            self.next_state(4)
        elif user_input == 2:
            print("Sorry!")
            self.next_state(1)
        elif user_input == 3:
            self.next_state(1)
        
    def state_8(self):
        """
        Menu n°8: Automatic tests menu
        """
        print("")
        print("*** Automatic tests menu - What do you want to do ? ***")
        print("(1) Create test images")
        print("(2) Start automatic tests")
        print("(3) Back to main menu")
        
        user_input = self.read_int([1,2,3])
        if user_input == 1:
            self.automatic_tests_state = 0
            self.next_state(9)
        elif user_input == 2:
            self.automatic_tests_state = 1
            self.next_state(9)
        elif user_input == 3:
            self.next_state(1)
     
    def state_9(self):
        """
        Menu n°9: Enter lower bound
        """
        print("")
        print("*** Please enter lower bound for automatic tests image's range ***")
        
        try:
            user_input = self.read_int(range(999))
        except:
            self.next_state(1)
            return
        self.automatic_tests_lower_bound = user_input
        self.next_state(10)
     
    def state_10(self):
        """
        Menu n°10: Enter upper bound
        """
        print("")
        print("*** Please enter upper bound for automatic tests image's range (not included) ***")
        
        try:
            user_input = self.read_int(range(999))
            while user_input <= self.automatic_tests_lower_bound:
                print(f"Please choose an upper bound bigger than your lower bound: {self.automatic_tests_lower_bound}")
                user_input = self.read_int(range(999))
        except:
            self.next_state(1)
            return

        self.automatic_tests_upper_bound = user_input
        self.next_state(11)
     
    def state_11(self):
        """
        Menu n°11: Enter operation
        """
        print("")
        print("*** Please choose an operation ***")
        print("(1) +")
        print("(2) -")
        print("(3) x")
        print("(4) /")
        print("(5) Back to main menu")
        
        user_input = self.read_int([1,2,3,4,5])
        if user_input == 1:
            self.automatic_tests_operator = "+"
            self.automatic_tests_subdirectory = "additions"
        elif user_input == 2:
            self.automatic_tests_operator = "-"
            self.automatic_tests_subdirectory = "soustractions"
        elif user_input == 3:
            self.automatic_tests_operator = "x"
            self.automatic_tests_subdirectory = "multiplications"
        elif user_input == 4:
            self.automatic_tests_operator = "/"
            self.automatic_tests_subdirectory = "divisions"
        elif user_input == 5:
            self.next_state(1)
            return
        
        if self.automatic_tests_state == 0:
            self.next_state(12)
        elif self.automatic_tests_state == 1:
            self.next_state(14)
     
    def state_12(self):
        """
        Menu n°11: Enter operation
        """
        print("")
        print("*** Please select an OpenCV font ***")
        print("(1) FONT_HERSHEY_SIMPLEX")
        print("(2) FONT_HERSHEY_PLAIN")
        print("(3) FONT_HERSHEY_DUPLEX")
        print("(4) FONT_HERSHEY_COMPLEX")
        print("(5) FONT_HERSHEY_TRIPLEX")
        print("(6) FONT_HERSHEY_COMPLEX_SMALL")
        print("(7) FONT_HERSHEY_SCRIPT_SIMPLEX")
        print("(8) FONT_HERSHEY_SCRIPT_COMPLEX")
        print("(9) FONT_ITALIC")
        print("(10) Back to main menu")
        
        user_input = self.read_int(range(10))
        if user_input == 1:
            self.automatic_tests_font = cv2.FONT_HERSHEY_SIMPLEX
        elif user_input == 2:
            self.automatic_tests_font = cv2.FONT_HERSHEY_PLAIN
        elif user_input == 3:
            self.automatic_tests_font = cv2.FONT_HERSHEY_DUPLEX
        elif user_input == 4:
            self.automatic_tests_font = cv2.FONT_HERSHEY_COMPLEX
        elif user_input == 5:
            self.automatic_tests_font = cv2.FONT_HERSHEY_TRIPLEX
        elif user_input == 6:
            self.automatic_tests_font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        elif user_input == 7:
            self.automatic_tests_font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        elif user_input == 8:
            self.automatic_tests_font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        elif user_input == 9:
            self.automatic_tests_font = cv2.FONT_ITALIC
        elif user_input == 10:
            self.next_state(1)
            return
        
        self.next_state(13)

    def state_13(self):
        """
        Menu n°13: Generate test images
        """
        print("")
        print("Generating test images. Please wait...")
        min_left_op = min_right_op = self.automatic_tests_lower_bound
        max_left_op = max_right_op = self.automatic_tests_upper_bound


        create_test_images(self.automatic_tests_subdirectory, min_left_op, min_right_op, max_left_op, max_right_op, self.automatic_tests_operator, self.automatic_tests_font)

        print("Test images have been created.")
        self.next_state(8)
     
    def state_14(self):
        """
        Menu n°14: Automatic testing
        """
        print("")
        print("Automatic testing. Please wait...")
        min_left_op = min_right_op = self.automatic_tests_lower_bound
        max_left_op = max_right_op = self.automatic_tests_upper_bound

        test_generated_operations(self.automatic_tests_subdirectory, self.model, min_left_op, min_right_op, max_left_op, max_right_op, self.automatic_tests_operator, )
        self.next_state(8)
        
    def next_state(self, state_id):
        """
        Changes the state.
        """
        self.state_id = state_id
        self.display_state()

    def display_state(self):
        """
        Displays current sate.
        """
        if self.state_id == 1:
            self.state_1()
        elif self.state_id == 3:
            self.state_3()
        elif self.state_id == 4:
            self.state_4()
        elif self.state_id == 5:
            self.state_5()
        elif self.state_id == 6:
            self.state_6()
        elif self.state_id == 7:
            self.state_7()
        elif self.state_id == 8:
            self.state_8()
        elif self.state_id == 9:
            self.state_9()
        elif self.state_id == 10:
            self.state_10()
        elif self.state_id == 11:
            self.state_11()
        elif self.state_id == 12:
            self.state_12()
        elif self.state_id == 13:
            self.state_13()
        elif self.state_id == 14:
            self.state_14()
        return