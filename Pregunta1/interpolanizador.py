import cv2 
import numpy

image_path = "C:/Users/Usuario/Desktop/ProyectoCN/Pregunta1/avion.jpg"  # Replace with your actual path
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]  
gray = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]


contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


largest_contour = None
largest_contour_area = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > largest_contour_area:
        largest_contour_area = area
        largest_contour = cnt

all_points = largest_contour.reshape(-1, 2)

# Estimate upper body height as a fraction of image height (adjust as needed)
upper_body_fraction = 0.6  # Experiment with this value

upper_body_height = int(image.shape[0] * upper_body_fraction)
upper_body_points = all_points[all_points[:, 1] < upper_body_height]


hull = cv2.convexHull(upper_body_points)

# Option 1: Spline interpolation
spline = cv2.approxPolyDP(hull, epsilon=0.01 * cv2.arcLength(hull, True), closed=True)

# Option 2: Bezier curve fitting (more complex)
# ... implement Bezier curve fitting based on upper_body_points

cv2.drawContours(image, [hull], -1, (0, 255, 0), 2)  # Green color for interpolated outline
cv2.imshow('Animal Silhouette with Interpolated Upper Body', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
