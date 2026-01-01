import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_color_features(hsv_contour, mask):
    """Extract mean HSV color features"""
    mean_h = np.mean(hsv_contour[mask > 0, 0])
    mean_s = np.mean(hsv_contour[mask > 0, 1])
    mean_v = np.mean(hsv_contour[mask > 0, 2])
    return [mean_h, mean_s, mean_v]

def extract_shape_features(contour):
    """Extract geometric features"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0: return [0,0,0,0,0]
    
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    x,y,w,h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    size_cat = 0 if area < 1000 else 1 if area < 5000 else 2
    return [area, perimeter, circularity, aspect_ratio, size_cat]

def extract_ripeness_feature(hsv_contour, mask):
    """Ripeness = bright + saturated pixels"""
    ripe = np.sum((hsv_contour[mask>0,1]>100) & (hsv_contour[mask>0,2]>150))
    total = np.sum(mask > 0)
    return ripe/total if total > 0 else 0

def get_all_features(image, contour, hsv):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    color = extract_color_features(hsv, mask)
    shape = extract_shape_features(contour)
    ripe = extract_ripeness_feature(hsv, mask)
    return np.array(color + shape + [ripe])

def preprocess_image(image):
    """Color segmentation like face detection"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    apple = cv2.inRange(hsv, np.array([0,50,50]), np.array([20,255,255]))
    orange = cv2.inRange(hsv, np.array([10,100,100]), np.array([25,255,255]))
    banana = cv2.inRange(hsv, np.array([20,100,100]), np.array([40,255,255]))
    mask = cv2.bitwise_or(apple, cv2.bitwise_or(orange, banana))
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return hsv, mask
