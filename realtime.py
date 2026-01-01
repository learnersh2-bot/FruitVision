import cv2
import numpy as np
import pickle
from features import preprocess_image, get_all_features

data = pickle.load(open('fruit_classifier.pkl', 'rb'))
model, scaler = data['model'], data['scaler']

cap = cv2.VideoCapture(0)
print("🔴 LIVE MODE - Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    orig = frame.copy()
    hsv, mask = preprocess_image(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < 500: continue
        
        feats = get_all_features(frame, contour, hsv)
        pred = model.predict(scaler.transform([feats]))[0]
        conf = model.predict_proba(scaler.transform([feats]))[0].max()
        
        if conf > 0.7:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(orig, (x,y), (x+w,y+h), (0,255,0), 3)
            cv2.putText(orig, f"{pred} {conf:.0%}", (x,y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    cv2.putText(orig, "LIVE FRUIT DETECTION - 'q' to quit", (10,30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow('🔴 LIVE FRUIT DETECTOR', orig)
    
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
