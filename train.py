import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler   # ← FIXED: Added import!
from features import preprocess_image, get_all_features
import matplotlib.pyplot as plt

def load_dataset(data_dir):
    fruits = ['apples', 'bananas', 'oranges']
    X, y, class_counts = [], [], {}
    
    for fruit_idx, fruit_name in enumerate(fruits):
        fruit_path = os.path.join(data_dir, fruit_name)
        class_counts[fruit_name] = 0
        
        if not os.path.exists(fruit_path):
            print(f"⚠️  Create {fruit_path} with 20+ images")
            continue
            
        files = [f for f in os.listdir(fruit_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        print(f"📁 {fruit_name}: {len(files)} images")
        
        for file in files:
            img_path = os.path.join(fruit_path, file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            hsv, mask = preprocess_image(img)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(cnt) > 200:
                    feats = get_all_features(img, cnt, hsv)
                    X.append(feats)
                    y.append(fruit_idx)
                    class_counts[fruit_name] += 1
    
    return np.array(X), np.array(y), class_counts

def train_model():
    print("🍎 Loading training data...")
    X, y, counts = load_dataset("data")
    
    print(f"\n📊 Dataset summary:")
    for fruit, count in counts.items():
        print(f"   {fruit}: {count} samples")
    
    if len(X) < 15:
        print("❌ Need 20+ images PER fruit folder!")
        return
    
    # Split + Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc:.1%} ({len(y_test)} test samples)")
    
    # ✅ FIXED CLASSIFICATION REPORT
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    names = ['🍎 Apple', '🍌 Banana', '🍊 Orange']
    report_labels = [names[i] for i in unique_labels]
    
    print("\n📈 Report:")
    print(classification_report(y_test, y_pred, labels=unique_labels, 
                               target_names=report_labels, zero_division=0))
    
    # Save
    pickle.dump({'model': model, 'scaler': scaler}, open('fruit_classifier.pkl', 'wb'))
    print("💾 Model saved!")
    
    # Plot
    plt.figure(figsize=(10,6))
    imp = model.feature_importances_
    feats = ['Hue','Sat','Val','Area','Perim','Circ','AspRat','Size','Ripe']
    plt.barh(feats, imp)
    plt.xlabel('Importance'); plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('importance.png', dpi=150)
    plt.show()
    
    print("\n✅ SUCCESS! Run 'python detect.py'")

if __name__ == "__main__":
    train_model()
