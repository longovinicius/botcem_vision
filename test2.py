import math

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

# Test data
regions = [(409, 335, 426, 351), (652, 316, 667, 331), (656, 314, 670, 328)]
filtered_regions = eliminate_close_regions(regions)

print(filtered_regions)
