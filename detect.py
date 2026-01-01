import cv2
import numpy as np
import pickle
import os
from features import preprocess_image, get_all_features

NAMES = ['🍎 Apple', '🍌 Banana', '🍊 Orange']
BOX_COLORS = [(36,255,12), (255,193,7), (255,99,71)]

def load_model():
    try:
        data = pickle.load(open('fruit_classifier.pkl', 'rb'))
        return data['model'], data['scaler']
    except:
        print("❌ Run 'python train.py' first!")
        return None, None

def process_image(img_path):
    model, scaler = load_model()
    if not model: return
    
    img = cv2.imread(img_path)
    orig = img.copy()
    hsv, mask = preprocess_image(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fruits_found = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 400: continue
        
        feats = get_all_features(img, contour, hsv)
        pred = model.predict(scaler.transform([feats]))[0]
        conf = model.predict_proba(scaler.transform([feats]))[0].max()
        
        if conf > 0.65:
            fruits_found += 1
            x,y,w,h = cv2.boundingRect(contour)
            
            # Box + label background
            cv2.rectangle(orig, (x,y), (x+w,y+h), BOX_COLORS[pred], 3)
            label = f"{NAMES[pred]} {conf:.0%}"
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(orig, (x,y-th-10), (x+tw+10,y), BOX_COLORS[pred], -1)
            cv2.putText(orig, label, (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    print(f"🎉 Found {fruits_found} fruits!")
    cv2.imshow('🍓 FRUIT DETECTOR', orig)
    cv2.waitKey(0); cv2.destroyAllWindows()
    cv2.imwrite('DETECTION_RESULT.jpg', orig)

if __name__ == "__main__":
    process_image('test_images/sample.jpg')
