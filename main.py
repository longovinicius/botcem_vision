import cv2
import numpy as np
import utils 

detected_colors = []
image_path = "full.jpg"
img = cv2.imread(image_path)
diag_size = utils.get_measurement(img)
while(True):

    average_color = utils.extract_color_from_region(img)
    print("Average Color (RGB):", average_color)
    region_coordinates = utils.find_color_regions(img, average_color, target_diagonal_size=diag_size)
    print("Region Coordinates:", region_coordinates)
    for region in region_coordinates:
        utils.show_region_in_image(img, region)
    detected_colors.append(region_coordinates)

    user_input = input("Press '+' to iterate again, or '-' to stop: ")
    if user_input == '+':
        # Put your code to execute when the user presses '+' here
        print("Looping...")
    elif user_input == '-':
        print("Loop stopped.")
        break
    else:
        print("Invalid input. Please try again.")

for regions in detected_colors:
    filtered_regions = utils.eliminate_close_regions(regions)
    print(filtered_regions)

#utils.show_region_in_image(image_path, region_coordinates)


