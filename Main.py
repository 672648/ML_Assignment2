import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, mean_squared_error
from PIL import Image

import os
#base_path trenger bildene den er trent på hvis du skal trene den selv last ned https://www.kaggle.com/datasets/snmahsa/human-images-dataset-men-and-women/data
#og sett base_path = hvor den ble lastet ned
#base_path = r'C:\Users\Jonas\PycharmProjects\oblig4ML\image'

for category in ['men', 'women']:
    category_path = os.path.join(base_path, category)
    for filename in os.listdir(category_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(category_path, filename)


men_folder = 'image/men'
women_folder = 'image/women'

data = []

for filename in os.listdir(men_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(men_folder, filename)
        data.append([image_path, 0])  # 0 for menn


for filename in os.listdir(women_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(women_folder, filename)
        data.append([image_path, 1])  # 1 for kvinner


df = pd.DataFrame(data, columns=['image_path', 'label'])


output_csv_path = 'gender_labels.csv'
df.to_csv(output_csv_path, index=False)

csv_path = 'gender_labels.csv'
df = pd.read_csv(csv_path)

images = []
labels = []
print(df.shape[0])

for index, row in df.iterrows():
    img_path = row['image_path']

    try:
        img = Image.open(img_path)
        img = img.resize((64, 64))
        img = np.array(img)
        if img.shape == (64, 64, 3):
            images.append(img.flatten())
            labels.append(row['label'])
    except Exception as e:
        print(f"Could not process image {img_path}: {e}")

images = np.array(images)
labels = np.array(labels)

images = images.astype('float32') / 255.0
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=41)
model = SVC(probability=True)
model.fit(X_train, y_train)
joblib.dump(model, 'gender_model.pkl')
y_pred = model.predict(X_val)
print(y_pred)
accuracy = model.score(X_val, y_val)

mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error (MSE): {mse}")


print(f'Valideringsnøyaktighet: {accuracy*100}%')

cm = confusion_matrix(y_val, y_pred)
print(f"Confusion Matrix:\n{cm}")

TN, FP, FN, TP = cm.ravel()

print(f'True Positives (TP): {TP}')
print(f'True Negatives (TN): {TN}')
print(f'False Positives (FP): {FP}')
print(f'False Negatives (FN): {FN}')
y_pred_proba = model.predict_proba(X_val)[:, 1]


fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

for i in range(len(thresholds)):
    print(f"Threshold: {thresholds[i]:.2f}, TPR (Recall): {tpr[i]:.2f}, FPR: {fpr[i]:.2f}")

