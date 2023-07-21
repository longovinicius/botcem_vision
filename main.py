import cv2
import numpy as np
import utils 

detected_colors = []
image_path = "real.jpg"
img = cv2.imread(image_path)
diag_size = utils.get_measurement(img)
while(True):

    average_color = utils.extract_non_black_color(img)
    print("Average Color (RGB):", average_color)
    contours = utils.get_contour(img, average_color, show_img=False)
    contour_data = utils.filtered_contour_and_center(contours, diag_size, visualize=True, img=img)
    detected_colors.append(contour_data)
    utils.print_dots(img, contour_data[1])

    user_input = input("Press '+' to iterate again, or '-' to stop: ")
    if user_input == '+':
        # Put your code to execute when the user presses '+' here
        print("Looping...")
    elif user_input == '-':
        print("Loop stopped.")
        break
    else:
        print("Invalid input. Please try again.")

for data in detected_colors:
    print(data)

