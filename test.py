import math

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def is_close_to_edge(region1, region2, threshold_distance=10):
    x1_r1, y1_r1, x2_r1, y2_r1 = region1
    x1_r2, y1_r2, x2_r2, y2_r2 = region2

    distance_x = min(euclidean_distance(x1_r1, y1_r1, x1_r2, y1_r2), euclidean_distance(x1_r1, y1_r1, x2_r2, y2_r2))
    distance_y = min(euclidean_distance(x2_r1, y2_r1, x1_r2, y1_r2), euclidean_distance(x2_r1, y2_r1, x2_r2, y2_r2))

    return distance_x <= threshold_distance or distance_y <= threshold_distance

def find_close_regions(regions_list, threshold_distance=10):
    close_regions_list = []

    for i, region1 in enumerate(regions_list):
        for j, region2 in enumerate(regions_list):
            if i != j and is_close_to_edge(region1[0], region2[0], threshold_distance):
                close_regions_list.append(region1[0])
                break

    return close_regions_list

def calculate_angle(region1, region2):
    x1_r1, y1_r1, x2_r1, y2_r1 = region1
    x1_r2, y1_r2, x2_r2, y2_r2 = region2

    center_r1 = ((x1_r1 + x2_r1) / 2, (y1_r1 + y2_r1) / 2)
    center_r2 = ((x1_r2 + x2_r2) / 2, (y1_r2 + y2_r2) / 2)

    vector_r1 = (x2_r1 - x1_r1, y2_r1 - y1_r1)
    vector_r2 = (x2_r2 - x1_r2, y2_r2 - y1_r2)

    dot_product = vector_r1[0] * vector_r2[0] + vector_r1[1] * vector_r2[1]
    magnitude_r1 = euclidean_distance(x1_r1, y1_r1, x2_r1, y2_r1)
    magnitude_r2 = euclidean_distance(x1_r2, y1_r2, x2_r2, y2_r2)

    angle_rad = math.acos(dot_product / (magnitude_r1 * magnitude_r2))
    angle_deg = math.degrees(angle_rad)

    return angle_deg
# Example list of regions
regions_list = [
    [(431, 210, 443, 224)],
    [(423, 202, 437, 215)],
    [(614, 275, 600, 153)],
    [(575, 137, 595, 151)],
    # Add more regions here...
]

close_regions = find_close_regions(regions_list)

print("Coordinates of close regions:")
for region in close_regions:
    print(region)
    

print("Coordinates of close regions:")
for i, region1 in enumerate(close_regions):
    for j, region2 in enumerate(close_regions):
        if i < j:  # Ensures we only calculate the angle once for each pair
            angle = calculate_angle(region1, region2)
            print(f"Angle between regions {i + 1} and {j + 1}: {angle} degrees")