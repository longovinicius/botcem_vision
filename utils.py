import cv2
import numpy as np
import math

def detect_template(image, tmp):
    main_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform template matching
    res = cv2.matchTemplate(main_gray, tmp, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8  # You can adjust this threshold based on the level of confidence you want

    # Get the coordinates of the matched region with the highest correlation score
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + tmp.shape[1], top_left[1] + tmp.shape[0])
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle

        # Show the main image with the detected pattern
        cv2.imshow('Detected Pattern', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print the coordinate of the detected pattern
        print(f"Pattern found at coordinate: ({top_left[0]}, {top_left[1]})")
        return top_left
    else:
        print("Pattern not found.")
        return None

def extract_color_from_region(image):

    # Function to handle mouse events
    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked, roi_x1, roi_y1, roi_x2, roi_y2, roi_hist
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = True
            roi_x1, roi_y1 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            clicked = False
            roi_x2, roi_y2 = x, y
            roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_hist = cv2.mean(roi)[:3]

    # Initialize variables
    clicked = False
    roi_x1, roi_y1, roi_x2, roi_y2 = -1, -1, -1, -1
    roi_hist = None

    # Create a window to display the image
    cv2.namedWindow("Extract_Color")
    cv2.setMouseCallback("Extract_Color", mouse_callback)

    while True:
        # Display the image
        cv2.imshow("Extract_Color", image)

        # If the region is selected, display the average color
        if clicked:
            roi_temp = image.copy()
            cv2.rectangle(roi_temp, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            cv2.imshow("Extract_Color", roi_temp)

        # Wait for a key press (ESC to exit)
        key = cv2.waitKey(1)
        if key == 27:
            break

    # Close the window
    cv2.destroyAllWindows()

    # Return the average color in RGB format
    if roi_hist is not None:
        average_color_rgb = tuple(reversed(roi_hist))
        return average_color_rgb
    else:
        return None

def extract_non_black_color(image):
    while True:
        average_color = extract_color_from_region(image)
        if average_color != (0, 0, 0):
            return average_color
        print("Got black, try again!")

def get_measurement(image):

    # Function to handle mouse events
    def mouse_callback(event, x, y, flags, param):
        nonlocal point1, point2, distance
        if event == cv2.EVENT_LBUTTONDOWN:
            if point1 is None:
                point1 = (x, y)
            else:
                point2 = (x, y)
                distance = np.linalg.norm(np.array(point1) - np.array(point2))

    # Initialize variables
    point1, point2 = None, None
    distance = None

    # Create a window to display the image
    cv2.namedWindow("Extract_measurement")
    cv2.setMouseCallback("Extract_measurement", mouse_callback)

    while True:
        # Display the image
        cv2.imshow("Extract_measurement", image)

        # If both points are selected, draw a line between them
        if point1 is not None and point2 is not None:
            cv2.line(image, point1, point2, (0, 255, 0), 2)

        # If a distance is calculated, display it as text
        if distance is not None:
            cv2.putText(image, f"{distance:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Wait for a key press (ESC to exit)
        key = cv2.waitKey(1)
        if key == 27:
            break

    # Close the window
    cv2.destroyAllWindows()

    # Return the distance in pixels
    return distance

def find_color_region(image, target_color_rgb, tolerance=30):

    # Convert target color to BGR format
    target_color_bgr = tuple(reversed(target_color_rgb))

    # Compute lower and upper bounds for color tolerance
    lower_bound = np.array([max(0, c - tolerance) for c in target_color_bgr], dtype=np.uint8)
    upper_bound = np.array([min(255, c + tolerance) for c in target_color_bgr], dtype=np.uint8)

    # Create a mask for the target color region
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (region with the target color)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Return the coordinates of the region where the average color is located
    return (x, y, x + w, y + h)

import cv2
import numpy as np

def find_color_regions(image, target_color_rgb, tolerance=30, target_diagonal_size=None):
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Can't read the image")

    # Convert target color to BGR format
    target_color_bgr = tuple(reversed(target_color_rgb))

    # Compute lower and upper bounds for color tolerance
    lower_bound = np.array([max(0, c - tolerance) for c in target_color_bgr], dtype=np.uint8)
    upper_bound = np.array([min(255, c + tolerance) for c in target_color_bgr], dtype=np.uint8)

    # Create a mask for the target color region
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize an empty list to store the region coordinates
    region_coordinates_list = []

    # Loop through all contours
    for contour in contours:
        # Calculate the minimum area rectangle that encloses the contour
        rect = cv2.minAreaRect(contour)

        # Get the center, width, and height of the rectangle
        center_x, center_y = map(int, rect[0])
        width, height = map(int, rect[1])

        # Calculate the diagonal size of the region
        diagonal_size = np.sqrt(width**2 + height**2)

        # Check if the diagonal size is within the specified range
        if target_diagonal_size * 0.5 <= diagonal_size <= target_diagonal_size * 1.5:
            # Calculate the distance of the center from the origin (0, 0)
            distance_from_origin = np.sqrt(center_x**2 + center_y**2)

            # Calculate the threshold distance (target_diagonal_size/2)
            threshold_distance = target_diagonal_size / 2

            # Check if the distance is greater than the threshold
            if distance_from_origin > threshold_distance:
                region_coordinates_list.append(
                    (int(center_x - width/2), int(center_y - height/2),
                     int(center_x + width/2), int(center_y + height/2))
                )

    # Return the list of coordinates of all regions matching the target color and diagonal size
    return region_coordinates_list

def get_contour(image, target_color_rgb, tolerance=30, show_img=False):

    # Convert target color to BGR format
    target_color_bgr = tuple(reversed(target_color_rgb))

    # Compute lower and upper bounds for color tolerance
    lower_bound = np.array([max(0, c - tolerance) for c in target_color_bgr], dtype=np.uint8)
    upper_bound = np.array([min(255, c + tolerance) for c in target_color_bgr], dtype=np.uint8)

    # Create a mask for the target color region
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if show_img:
        img_with_contours = image.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Contours", img_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return contours

def filtered_contour_and_center(contours, pixel_size, visualize=False, img=None):
    filtered_contours = []
    center_points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > (pixel_size / 2) ** 2:
            x, y, w, h = cv2.boundingRect(contour)
            x_min, y_min = x, y
            x_max, y_max = x + w, y + h
            distance = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
            if distance < pixel_size * 2:
                filtered_contours.append(contour)
                center_points.append((int(x_min + w/2), int(y_min + h/2)))

    if visualize:
        img_with_contours = cv2.drawContours(img.copy(), filtered_contours, -1, (0, 255, 0), 2)
        cv2.imshow("Contours", img_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return filtered_contours, center_points

def print_dots(image, points, color=(0, 0, 255), radius=2, thickness=-1):
    # Draw circles (dots) at the specified points on the image
    for point in points:
        cv2.circle(image, point, radius, color, thickness)

def get_angle_between_points(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# def find_color_regions(image, target_color_rgb, tolerance=30, target_diagonal_size=None):
#     # Check if the image was loaded successfully
#     if image is None:
#         raise ValueError(f"Cant read image")

#     # Convert target color to BGR format
#     target_color_bgr = tuple(reversed(target_color_rgb))

#     # Compute lower and upper bounds for color tolerance
#     lower_bound = np.array([max(0, c - tolerance) for c in target_color_bgr], dtype=np.uint8)
#     upper_bound = np.array([min(255, c + tolerance) for c in target_color_bgr], dtype=np.uint8)

#     # Create a mask for the target color region
#     mask = cv2.inRange(image, lower_bound, upper_bound)

#     # Find contours in the mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Initialize an empty list to store the region coordinates
#     region_coordinates_list = []

#     # Loop through all contours and find their bounding rectangles
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)

#         # Calculate the diagonal size of the region
#         diagonal_size = np.sqrt(w**2 + h**2)

#         # Check if the diagonal size is within the specified range
#         if target_diagonal_size * 0.5 <= diagonal_size <= target_diagonal_size * 1.5:
#             # Calculate the center of the region
#             center_x, center_y = x + w // 2, y + h // 2

#             # Calculate the distance of the center from the origin (0, 0)
#             distance_from_origin = np.sqrt(center_x**2 + center_y**2)

#             # Calculate the threshold distance (target_diagonal_size/3)
#             threshold_distance = target_diagonal_size / 2

#             # Check if the distance is greater than the threshold
#             if distance_from_origin > threshold_distance:
#                 region_coordinates_list.append((x, y, x + w, y + h))

#     # Return the list of coordinates of all regions matching the target color and diagonal size
#     return region_coordinates_list

def show_region_in_image(image, region_coordinates):
    # Extract region coordinates
    x1, y1, x2, y2 = region_coordinates

    # Draw a rectangle around the specified region
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the image with the region highlighted
    cv2.imshow("Image with Region", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_area(region):
    x1, y1, x2, y2 = region
    return abs((x2 - x1) * (y2 - y1))

def eliminate_close_regions(coordinates, threshold=10):
    to_remove = set()
    for i, region in enumerate(coordinates):
        for j, other_region in enumerate(coordinates):
            if i != j:  # Avoid comparing a region with itself
                distance = euclidean_distance((region[0], region[1]), (other_region[0], other_region[1]))
                if distance < threshold:
                    # Determine which region is smaller and add its index to the to_remove set
                    area_i = calculate_area(region)
                    area_j = calculate_area(other_region)
                    if area_i < area_j:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
    
    result = [region for i, region in enumerate(coordinates) if i not in to_remove]
    return result
    
