import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.feature import hog

# 1. Get Data: Load the Olivetti faces dataset (400 images of 40 distinct people)
# This might take a moment to download the first time
print("Loading dataset...")
data = datasets.fetch_olivetti_faces()
images = data.images
targets = data.target

print(f"Dataset loaded. Images shape: {images.shape}")

# 2. Extract HOG Features
# We cannot feed raw pixels to an SVM efficiently. We feed it HOG features.
hog_features = []
print("Extracting HOG features from all images...")

i = 0
for image in images:
    # show images after every 50 images along side HOG images for comparison
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(1, 1), visualize=True)
    if i % 50 == 0:
        print(f"Index : {i}", sep=None)

        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        #ax1.axis('off') 
        #ax1.imshow(image, cmap=plt.cm.gray)
        #ax1.set_title('Input image')
        #ax2.axis('off')
        #ax2.imshow(hog_image, cmap=plt.cm.gray)
        #ax2.set_title('Histogram of Oriented Gradients')
        #plt.show()
        
    hog_features.append(fd)
    i += 1

X = np.array(hog_features)
y = targets

# 3. Split Data
# Train on 80% of the faces, Test on the remaining 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the SVM
# We use a Linear Kernel because HOG features are usually linearly separable
print("Training SVM classifier...")
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# 5. Predict on the Test Set
print("Predicting on test set...")
y_pred = clf.predict(X_test)

# 6. Evaluate
print(f"\nAccuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")

# 7. Visualize a Prediction
# Let's grab a random test image and see if the SVM gets it right
index_to_show = 5
test_img_feature = X_test[index_to_show].reshape(1, -1)
prediction = clf.predict(test_img_feature)[0]
actual = y_test[index_to_show]

# Find the original image corresponding to this test feature
# (A bit tricky since we split the arrays, but for demo we just show the result text)
print(f"--- Example ---")
print(f"SVM Predicted Person ID: {prediction}")
print(f"Actual Person ID:        {actual}")
print(f"Result: {'CORRECT' if prediction == actual else 'WRONG'}")

# Optional: Visualize the first few predictions
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for ax, idx in zip(axes, range(4)):
    # We can't easily show the image here because X_test only has HOG data,
    # but we can verify the math works.
    pred = clf.predict(X_test[idx].reshape(1, -1))[0]
    act = y_test[idx]
    ax.text(0.5, 0.5, f"Pred: {pred}\nActual: {act}", 
            ha='center', va='center', fontsize=12)
    ax.axis('off')
    ax.set_title(f"{'Correct' if pred == act else 'Wrong'}")
plt.show()

# Load an image zidance.jpg and test with the classifier
from skimage import io
img = io.imread('zidane.jpg', as_gray=True)
img = cv2.resize(img, (64, 64))
print(f"Image shape: {img.shape}")
plt.imshow(img, cmap=plt.cm.gray)
plt.show()
img = img / 255.0

fd, hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True)
prediction = clf.predict(fd.reshape(1, -1))

# Display the result with side by side with raw image. 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title(f'Prediction: {prediction}')
plt.show()  

# show prediction results with bounding box on raw image
import cv2
