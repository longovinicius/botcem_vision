import cv2
import numpy as np
import math

class ImageProcessor:
    def __init__(self, img):
        self.distance = None
        self.img = img

    def get_measurement(self):
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
            cv2.imshow("Extract_measurement", self.img)

            # If both points are selected, draw a line between them
            if point1 is not None and point2 is not None:
                cv2.line(self.img, point1, point2, (0, 255, 0), 2)

            # If a distance is calculated, display it as text
            if distance is not None:
                cv2.putText(self.img, f"{distance:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Wait for a key press (ESC to exit)
            key = cv2.waitKey(1)
            if key == 27:
                break

    # Close the window
        cv2.destroyAllWindows()

        # Return the distance in pixels
        self.distance = distance
    

    def extract_color(self):

        # Function to handle mouse events
        def mouse_callback(event, x, y, flags, param):
            nonlocal clicked, roi_x1, roi_y1, roi_x2, roi_y2, roi_hist
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked = True
                roi_x1, roi_y1 = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                clicked = False
                roi_x2, roi_y2 = x, y
                roi = self.img[roi_y1:roi_y2, roi_x1:roi_x2]
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
            cv2.imshow("Extract_Color", self.img)

            # If the region is selected, display the average color
            if clicked:
                roi_temp = self.img.copy()
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

    def extract_non_black_color(self):
        while True:
            average_color = self.extract_color()
            if average_color != (0, 0, 0):
                return average_color
            print("Got black, try again!")


    def get_contour(self, target_color_rgb, tolerance=30, show_img=False):

        # Convert target color to BGR format
        target_color_bgr = tuple(reversed(target_color_rgb))

        # Compute lower and upper bounds for color tolerance
        lower_bound = np.array([max(0, c - tolerance) for c in target_color_bgr], dtype=np.uint8)
        upper_bound = np.array([min(255, c + tolerance) for c in target_color_bgr], dtype=np.uint8)

        # Create a mask for the target color region
        mask = cv2.inRange(self.img, lower_bound, upper_bound)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if show_img:
            img_with_contours = self.img.copy()
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

    def print_dots(self, points, color=(0, 0, 255), radius=2, thickness=-1):
    # Draw circles (dots) at the specified points on the image
        for point in points:
            cv2.circle(self.img, point, radius, color, thickness)

    def get_angle_between_points(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

# Example usage:
# if __name__ == "__main__":
#     
# Load an image
    # img = cv2.imread("real.jpg")

    # vision = ImageProcessor(img)

    # # Call the functions using the instance of the class
    # distance = vision.get_measurement()

    # average_color = vision.extract_non_black_color()

    # target_color_rgb = (0, 255, 0)  # Example target color (green)
    # contours = vision.get_contour(target_color_rgb)