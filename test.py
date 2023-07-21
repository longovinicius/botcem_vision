import math

def find_close_coordinates(coordinates, distance_threshold):
    close_tuples = []

    for i, (x1, y1) in enumerate(coordinates):
        for x2, y2 in coordinates[i + 1:]:
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if distance <= distance_threshold:
                close_tuples.append((x1, y1, x2, y2))

    return close_tuples

def calculate_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# Example usage:
coordinates = [(1, 2), (4, 5), (2, 3), (10, 12), (9, 8)]
distance_threshold = 3  # Adjust this value based on your requirement

close_tuples = find_close_coordinates(coordinates, distance_threshold)
for x1, y1, x2, y2 in close_tuples:
    angle = calculate_angle(x1, y1, x2, y2)
    print(f"Angle between ({x1}, {y1}) and ({x2}, {y2}): {angle} degrees")
