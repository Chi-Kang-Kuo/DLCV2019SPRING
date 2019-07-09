import os
import cv2
import numpy as np
import matplotlib.pyplot as plt



# load data

data_path = './p2_data'
output_path = './output_image'

if not os.path.exists(output_path):
    os.makedirs(output_path)

train_image, test_image = [], []
train_label, test_label = [], []

for i in range(40):
        for j in range(10):
            image_name = os.path.join(data_path, str(i+1) + "_" + str(j+1) + ".png")
            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            if j < 6:
                train_image.append(image.flatten())
                train_label.append(i+1)
            else:
                test_image.append(image.flatten())
                test_label.append(i+1)
                
train_image, train_label = np.array(train_image), np.array(train_label)
test_image, test_label = np.array(test_image), np.array(test_label)

# train_image.shape = (240, 2576)
# test_image.shape = (160, 2576)

# normalize (mean scale to zero)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_std=False)
scaler.fit(train_image) #shape of data = [n_samples, n_features]
scaled_data = scaler.transform(train_image)
 
# scaled_data.shape = (240, 2576)

# mean face

fig, ax = plt.subplots(1, 1, figsize=(3,3))
ax.imshow(scaler.mean_.reshape(56,46),cmap='gray')
ax.set_title("mean face")
fig.savefig(os.path.join(output_path, 'mean_face.png'))



# principal component analysis by sklearn
from sklearn.decomposition import PCA

pca = PCA(n_components=239)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)


# eigen_faces
fig, axes = plt.subplots(1, 4, figsize=(12,3))
for i in range(4):
    
    axes[i].imshow(pca.components_[i].reshape(56,46),cmap='gray')
    order = ["1st","2nd","3rd","4th"]
    axes[i].set_title(order[i]+" eigenface")

fig.savefig(os.path.join(output_path, 'eigen_faces.png'))


#original image
img1 = cv2.imread('p2_data/1_1.png',cv2.IMREAD_GRAYSCALE)
fig, ax = plt.subplots(1, 1, figsize=(3,3))
ax.imshow(img1,cmap='gray')
ax.set_title("person_1_image_1")
fig.savefig(os.path.join(output_path,'person_1_image_1.png'))

# reconstruction n = 3, 45, 140, 229
n_eigenfaces = [3,45,140,229]
fig, axes = plt.subplots(1, 4, figsize=(12,4))
for index,n in enumerate(n_eigenfaces):
    eigen_list = [i for i in range(n)]
    img = (np.dot(x_pca[[0],eigen_list],pca.components_[eigen_list,:])+scaler.mean_).reshape(56,46)
    mse = (np.square(img - img1)).mean()
    axes[index].imshow(img,cmap='gray')
    axes[index].set_title("using the first \n " + str(n) + " eigenfaces")
    axes[index].set_xlabel("mse: " + str(mse),fontsize=12)
fig.savefig(os.path.join(output_path, 'reconstructed_images.png'))


# Apply knn to testing images

#normalize testing data (minus train.mean)
scaled_test = scaler.transform(test_image)

test_pca = pca.transform(scaled_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix

n_range = [3,45,140]
k_range = [1,3,5]

# n eigenfaces
for n in n_range:
    
    eigen_list = [i for i in range(n)]
    train_knn = x_pca[:,eigen_list]
    test_knn = test_pca[:,eigen_list]
    
    # k nearest neighbors
    for k in k_range:
        
        knn = KNeighborsClassifier(n_neighbors=k)
        
        scores = cross_val_score(knn, train_knn, train_label, cv=3, scoring='accuracy')
        print('%d eigenfaces %d nearest neighbors' %(n,k))
        print(scores.mean())

#choose n = 45, k = 1

n, k = 45, 1
print('choose n = {}, k = {}'.format(n,k))
eigen_list = [i for i in range(n)]
train_knn = x_pca[:,eigen_list]
test_knn = test_pca[:,eigen_list]

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(train_knn,train_label)
pred = knn.predict(test_knn)
print('%d eigenfaces %d nearest neighbors' %(n,k))
print('confusion_matrix(label, predict)')
print(confusion_matrix(test_label,pred)) # 40 class
print('classification_report(label, predict)')
print(classification_report(test_label,pred))