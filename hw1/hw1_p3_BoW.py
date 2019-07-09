import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(46)


# load data

data_path = './p3_data'
output_path = './output_image'

if not os.path.exists(output_path):
    os.makedirs(output_path)

categories = sorted([category for category in os.listdir(data_path)])
train_image, test_image = [], []
train_label, test_label = [], []

# data preprocessing

for i, category in enumerate(categories):
    
    for j in range(1, 501): # each class has 500 imgs

        image_name = os.path.join(data_path, category, category + '_' + '{0:03}'.format(j) + '.JPEG')
        image = cv2.imread(image_name)

        if j < 376: # 375 * 4 = 1500
            train_image.append(image)
            train_label.append(i+1)
        else:       # 125 * 4 = 500
            test_image.append(image)
            test_label.append(i+1)

train_image, train_label = np.array(train_image), np.array(train_label)
test_image, test_label = np.array(test_image), np.array(test_label)

print('train_image.shape:', train_image.shape) # (1500, 64, 64, 3)
print('test_image.shape:', test_image.shape)   # (500, 64, 64, 3)

# images split into patches

from skimage.util.shape import view_as_blocks

block_shape = (16, 16, 3)

## plot 3 patches for one image from each category

for i in [0,375,750,1125]:
    B = view_as_blocks(train_image[i], block_shape)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9,3))
    for ax in axes:
    
        img_rgb = cv2.cvtColor(B[random.randint(0,3),random.randint(0,3),0], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
    fig.savefig(os.path.join(output_path,'bow_1_'+str(i)+'.png'))
    
    
train_image_patches = [] # train_image_patches = (16*1500,768)
test_image_patches = [] # test_image_patches = (16*500)

for i, image in enumerate(train_image):
    B = view_as_blocks(train_image[i], block_shape)
    for row in range(4):
        for col in range(4):
            train_image_patches.append(B.reshape(4,4,16,16,3)[row,col].reshape(768)) 
    
for i, image in enumerate(test_image):
    B = view_as_blocks(test_image[i], block_shape)
    for row in range(4):
        for col in range(4):
            test_image_patches.append(B.reshape(4,4,16,16,3)[row,col].reshape(768)) 

train_image_patches = np.array(train_image_patches)
test_image_patches= np.array(test_image_patches)

print('train_image_patches.shape:', train_image_patches.shape) # (16 * 1500, 768)
print('test_image_patches.shape:', test_image_patches.shape)   # (16 *  500, 768)

### show a particular patch
a_patch = train_image_patches[1].reshape(16,16,3)
a_patch = a_patch.astype('uint8')
img_rgb = cv2.cvtColor(a_patch, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
###

#### k-means to divide the training patches (features) into C clusters. 
#### You may choose C = 15 and maximum number of iterations = 5000 for simplicity.
print('Fit 24000 samples in 768 dims into 15 clusters.')

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=15,max_iter=5000)
print('Start kmeans......, it should take about 1 min')
kmeans.fit(train_image_patches)

print('shape of cluster centers:', kmeans.cluster_centers_.shape)
print('shape of labels:', kmeans.labels_.shape)

def ClusterIndices(clustNum, labels_array):
    return np.where(labels_array == clustNum)[0]

# show the sample indexes that clustered into a certain cluster
print('show the sample indexes that clustered into the 6th cluster')
print(ClusterIndices(6,kmeans.labels_))

#### perform PCA to reduce dimension from 768 to 3
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler(with_std=False)
scaler.fit(train_image_patches) #shape of data = [n_samples, n_features]
scaled_data = scaler.transform(train_image_patches)

pca = PCA(n_components=3)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

# random select 6 clusters to show on a 3D figure

selected_clusters = np.random.choice(15,size=6,replace=False)
print('random select 6 clusters:', selected_clusters)
# center of the 6 selected clusters
cent = pca.transform(scaler.transform(kmeans.cluster_centers_[selected_clusters,:]))

# 3D figures

from mpl_toolkits.mplot3d import Axes3D

X = x_pca[ClusterIndices(selected_clusters[0],kmeans.labels_),:]
label = np.ones(len(ClusterIndices(selected_clusters[0],kmeans.labels_)))*selected_clusters[0]
for i in selected_clusters:
    if i == selected_clusters[0]:
        continue
    X = np.concatenate((X,x_pca[ClusterIndices(i,kmeans.labels_),:]), axis=0)
    label = np.concatenate((label,np.ones(len(ClusterIndices(i,kmeans.labels_)))*i))
    
fig = plt.figure(figsize=(4, 4))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .9, 1], elev=48, azim=134)
plt.cla()
ax.scatter(cent[:, 0], cent[:, 1], cent[:, 2], cmap=plt.cm.get_cmap("Spectral"),marker='+', s=300, c='k')  
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=label, cmap=plt.cm.get_cmap("Spectral"),marker='.', s=0.1)
fig.savefig(os.path.join(output_path,'bow_2.png'))

#### represent each image as BoW features
### 3-3與3-4兩小題必須直接在768維下執行距離計算與k-NN分類，不應降維至3維。降至三維僅用於3-2小題的視覺化

# dictionary of visual words = 15 clusters (from k-means!!!)
# compute the Euclidean distances between each patch and 15 centers of clusters
# between kmeans.cluster_centers_ and train_image_patches
def l2norm_of_clusters(patch):
    distances = []
    for i in range(len(kmeans.cluster_centers_)):
        distances.append(np.linalg.norm(patch-kmeans.cluster_centers_[i]))
    return np.array(distances).reshape(15)

def normal_row_max_pool(x):
    for i in range(len(x)):
        x[i] = np.reciprocal(table_of_l2[i])/np.reciprocal(table_of_l2[i]).sum()
    result = []
    for i in range(len(x[0])):
        result.append(x[:,i].max())
    return np.array(result).reshape(15)

# train_image_patches (24000,768) to train_BoW = (1500,15)
train_BoW = []
for i in range(len(train_image)):
    
    table_of_l2 = []
    for patches in range(16):
        table_of_l2.append(l2norm_of_clusters(train_image_patches[i*16]))
        
    table_of_l2 = np.array(table_of_l2)
    train_BoW.append(normal_row_max_pool(table_of_l2))
    
train_BoW = np.array(train_BoW)

# test_image_patches (8000,768) to test_BoW = (500,15)
test_BoW = []
for i in range(len(test_image)):
    
    table_of_l2 = []
    for patches in range(16):
        table_of_l2.append(l2norm_of_clusters(test_image_patches[i*16]))
        
    table_of_l2 = np.array(table_of_l2)
    test_BoW.append(normal_row_max_pool(table_of_l2))
    
test_BoW = np.array(test_BoW)

print('train_BoW.shape:', train_BoW.shape)
print('test_BoW.shape:', test_BoW.shape)

images = [0,375,750,1125]
fig, axes = plt.subplots(1, 4, figsize=(12,4))
classes = ["banana","fountain","reef","tractor"]
for index,image in enumerate(images):
    axes[index].bar(np.arange(15),train_BoW[image])
    axes[index].set_title("an image from " + classes[index])
    axes[index].set_xlabel("Bag of Words",fontsize=12)
    
fig.savefig(os.path.join(output_path,'bow_3.png'))

#### Adopt the k-NN to perform classification on test_images using the above BoW features 
###  You may choose k = 5 for simplicity. Report the classification accuracy.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix

#train on train_BoW (1500, 15) label are the 4 classes
#the 15 centorids of the k-means clustering are the 15 BoW!! 
#train (1500,768)
#test on test_image_patches (8000, 768)
print('shape of labels of train_image_patches:', kmeans.labels_.shape)
print('shape of centers of 15 BoW:', kmeans.cluster_centers_.shape)

# k nearest neighbors

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_BoW,train_label)
pred = knn.predict(test_BoW)
print('classification report on test_images')
print(classification_report(test_label,pred)) 
