#!/usr/bin/env python
# coding: utf-8

# ### 1. IMPORTING LIBRARIES

# In[1]:


###IMPORTING LIBRARIES

import numpy as np

import os
import gzip

import mnist_reader

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import svm
from sklearn.svm import SVC


# ### 2. LOADING FASHION-MNIST DATASET FROM PATH

# In[2]:


###LOADING MNIST DATASET


def load_mnist(path, kind='train'):
   
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)

    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)

    print("Dataset Loaded")
    
    return images, labels


# ### 3. LOADING TEST AND TRAIN SET FEATURES AND LABELS

# In[3]:


###LOADING TRAIN AND TEST SET FEATURES AND LABELS

X_train, y_train = mnist_reader.load_mnist('C:/Users/evars/OneDrive/Desktop/ENEE633_P2', kind='train')
X_test, y_test = mnist_reader.load_mnist('C:/Users/evars/OneDrive/Desktop/ENEE633_P2', kind='t10k')


# ### 4. DATA NORMALIZING

# In[4]:


###NORMALIZING AND CHECKING THE SHAPES OF TRAIN AND TEST SETS



X_train = X_train/255

X_test = X_test/255

print("Feature Train and Test datasets are normalized")

#print(X_train[1])

#X_train = X_train.reshape(X_train.shape).T   

#y_train = y_train[np.newaxis]

#X_test = X_test.reshape(X_test.shape).T   

#y_test = y_test[np.newaxis]

print("Shape of Train set features (X_train) :  ",X_train.shape)
print("Shape of Train set labels (y_train) :  ",y_train.shape)
print("Shape of Test set features (X_test) :  ",X_test.shape)
print("Shape of Test set labels (y_test) :  ",y_test.shape)


# ### 5. APPLYING LDA TO TEST AND TRAIN

# In[5]:


###  LDA


lda = LinearDiscriminantAnalysis()

X_train_lda = lda.fit_transform(X_train,y_train)

X_test_lda = lda.transform(X_test)

print("Shape of Feature Test set Before LDA: ", X_train.shape)
print("Shape of Feature Test set After LDA: ", X_train_lda.shape)


print("Shape of Feature Test set Before LDA: ", X_test.shape)
print("Shape of Feature Test set After LDA: ", X_test_lda.shape)


# ### 5.1 Applying LDA to Linear SVM

# In[6]:


###SVM - Linear

svmclassifier = SVC(kernel='linear')
svmclassifier.fit(X_train_lda, y_train)

y_pred_svmlda = svmclassifier.predict(X_test_lda)

test_acc_lda = accuracy_score(y_test,y_pred_svmlda)*100

print("Test accuracy of Linear SVM After LDA: ",test_acc_lda)


# ### 5.2 Applying LDA to Kernel SVM (for 1, 2, 3, 5 and 8 degree polynomials) 

# In[7]:


svmclassifier = SVC(kernel='poly', degree=8)
svmclassifier.fit(X_train_lda, y_train)

y_pred_svmlda = svmclassifier.predict(X_test_lda)

test_acc_lda = accuracy_score(y_test,y_pred_svmlda)*100

print("Test accuracy of Kernel SVM of degree 8 After LDA: ",test_acc_lda)


# In[8]:


svmclassifier = SVC(kernel='poly', degree=5)
svmclassifier.fit(X_train_lda, y_train)

y_pred_svmlda = svmclassifier.predict(X_test_lda)

test_acc_lda = accuracy_score(y_test,y_pred_svmlda)*100

print("Test accuracy of Kernel SVM of degree 5 After LDA: ",test_acc_lda)


# In[9]:


svmclassifier = SVC(kernel='poly', degree=3)
svmclassifier.fit(X_train_lda, y_train)

y_pred_svmlda = svmclassifier.predict(X_test_lda)

test_acc_lda = accuracy_score(y_test,y_pred_svmlda)*100

print("Test accuracy of Kernel SVM of degree 3 After LDA: ",test_acc_lda)


# In[10]:


svmclassifier = SVC(kernel='poly', degree=2)
svmclassifier.fit(X_train_lda, y_train)

y_pred_svmlda = svmclassifier.predict(X_test_lda)

test_acc_lda = accuracy_score(y_test,y_pred_svmlda)*100

print("Test accuracy of Kernel SVM of degree 2 After LDA: ",test_acc_lda)


# In[11]:


svmclassifier = SVC(kernel='poly', degree=1)
svmclassifier.fit(X_train_lda, y_train)

y_pred_svmlda = svmclassifier.predict(X_test_lda)

test_acc_lda = accuracy_score(y_test,y_pred_svmlda)*100

print("Test accuracy of Kernel SVM of degree 1 After LDA: ",test_acc_lda)


# ### 5.3 Applying LDA to Radial Basis Function SVM Kernel

# In[12]:


svmclassifier = SVC(kernel='rbf')
svmclassifier.fit(X_train_lda, y_train)

y_pred_svmlda = svmclassifier.predict(X_test_lda)

test_acc_lda = accuracy_score(y_test,y_pred_svmlda)*100

print("Test accuracy of RBF SVM After LDA: ",test_acc_lda)


# ### 6. APPLYING PCA TO TEST AND TRAIN

# ### 6.1 PCA for n_components = 15

# In[13]:


### PCA

pca = PCA(n_components=15)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

print("Shape of Feature Test set Before PCA: ", X_train.shape)
print("Shape of Feature Test set After PCA for n_components = 15: ", X_train_pca.shape)


print("Shape of Feature Test set Before PCA: ", X_test.shape)
print("Shape of Feature Test set After PCA for n_components = 15: ", X_test_pca.shape)


# ### 6.1.1 Linear SVM

# In[14]:


###SVM - Linear

svmclassifier = SVC(kernel='linear')
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Linear SVM After PCA for n_components = 15: ",test_acc_pca)


# ### 6.1.2 Kernel SVM for polnomial degree 8, 5, 3, 2, 1

# In[15]:


svmclassifier = SVC(kernel='poly', degree=8)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 8 After PCA for n_components = 15: ",test_acc_pca)


# In[16]:


svmclassifier = SVC(kernel='poly', degree=5)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 5 After PCA for n_components = 15: ",test_acc_pca)


# In[17]:


svmclassifier = SVC(kernel='poly', degree=3)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 3 After PCA for n_components = 15: ",test_acc_pca)


# In[18]:


svmclassifier = SVC(kernel='poly', degree=2)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 2 After PCA for n_components = 15: ",test_acc_pca)


# In[19]:


svmclassifier = SVC(kernel='poly', degree=1)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 1 After PCA for n_components = 15: ",test_acc_pca)


# ### 6.1.3 RBF SVM

# In[20]:


svmclassifier = SVC(kernel='rbf')
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of RBF SVM After PCA for n_components = 15: ",test_acc_pca)


# ### 6.2 PCA for n_components = 30

# In[21]:


### PCA

pca = PCA(n_components=30)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

print("Shape of Feature Test set Before PCA: ", X_train.shape)
print("Shape of Feature Test set After PCA for n_components = 30: ", X_train_pca.shape)


print("Shape of Feature Test set Before PCA: ", X_test.shape)
print("Shape of Feature Test set After PCA for n_components = 30: ", X_test_pca.shape)


# ### 6.2.1 Linear SVM 

# In[22]:


###SVM - Linear

svmclassifier = SVC(kernel='linear')
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Linear SVM After PCA for n_components = 30: ",test_acc_pca)


# ### 6.2.2 Kernel SVM for polnomial degree 8, 5, 3, 2, 1

# In[23]:


svmclassifier = SVC(kernel='poly', degree=8)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 8 After PCA for n_components = 30: ",test_acc_pca)


# In[24]:


svmclassifier = SVC(kernel='poly', degree=5)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 5 After PCA for n_components = 30: ",test_acc_pca)


# In[25]:


svmclassifier = SVC(kernel='poly', degree=3)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 3 After PCA for n_components = 30: ",test_acc_pca)


# In[26]:


svmclassifier = SVC(kernel='poly', degree=2)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 2 After PCA for n_components = 30: ",test_acc_pca)


# In[27]:


svmclassifier = SVC(kernel='poly', degree=1)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 1 After PCA for n_components = 30: ",test_acc_pca)


# ### 6.2.3 RBF SVM

# In[28]:


svmclassifier = SVC(kernel='rbf')
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of RBF SVM After PCA for n_components = 30: ",test_acc_pca)


# ### 6.3 PCA for n_components = 50

# In[29]:


### PCA

pca = PCA(n_components=50)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

print("Shape of Feature Test set Before PCA: ", X_train.shape)
print("Shape of Feature Test set After PCA for n_components = 50: ", X_train_pca.shape)


print("Shape of Feature Test set Before PCA: ", X_test.shape)
print("Shape of Feature Test set After PCA for n_components = 50: ", X_test_pca.shape)


# ### 6.3.1 Linear SVM 

# In[30]:


###SVM - Linear

svmclassifier = SVC(kernel='linear')
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Linear SVM After PCA for n_components = 50: ",test_acc_pca)


# ### 6.3.2 Kernel SVM for polnomial degree 8, 5, 3, 2, 1

# In[31]:


svmclassifier = SVC(kernel='poly', degree=8)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 8 After PCA for n_components = 50: ",test_acc_pca)


# In[32]:


svmclassifier = SVC(kernel='poly', degree=5)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 5 After PCA for n_components = 50: ",test_acc_pca)


# In[33]:


svmclassifier = SVC(kernel='poly', degree=3)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 3 After PCA for n_components = 50: ",test_acc_pca)


# In[34]:


svmclassifier = SVC(kernel='poly', degree=2)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 2 After PCA for n_components = 50: ",test_acc_pca)


# In[35]:


svmclassifier = SVC(kernel='poly', degree=1)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 1 After PCA for n_components = 50: ",test_acc_pca)


# ### 6.3.3 RBF SVM

# In[36]:


svmclassifier = SVC(kernel='rbf')
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of RBF SVM After PCA for n_components = 50: ",test_acc_pca)


# ### 6.4 PCA for n_components = 75

# In[37]:


### PCA

pca = PCA(n_components=75)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

print("Shape of Feature Test set Before PCA: ", X_train.shape)
print("Shape of Feature Test set After PCA for n_components = 75: ", X_train_pca.shape)


print("Shape of Feature Test set Before PCA: ", X_test.shape)
print("Shape of Feature Test set After PCA for n_components = 75: ", X_test_pca.shape)


# ### 6.4.1 Linear SVM

# In[38]:


###SVM - Linear

svmclassifier = SVC(kernel='linear')
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Linear SVM After PCA for n_components = 75: ",test_acc_pca)


# ### 6.4.2 Kernel SVM for polnomial degree 8, 5, 3, 2, 1

# In[39]:


svmclassifier = SVC(kernel='poly', degree=8)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 8 After PCA for n_components = 75: ",test_acc_pca)


# In[40]:


svmclassifier = SVC(kernel='poly', degree=5)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 5 After PCA for n_components = 75: ",test_acc_pca)


# In[41]:


svmclassifier = SVC(kernel='poly', degree=3)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 3 After PCA for n_components = 75: ",test_acc_pca)


# In[42]:


svmclassifier = SVC(kernel='poly', degree=2)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 2 After PCA for n_components = 75: ",test_acc_pca)


# In[43]:


svmclassifier = SVC(kernel='poly', degree=1)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 1 After PCA for n_components = 75: ",test_acc_pca)


# ### 6.4.3 RBF SVM

# In[44]:


svmclassifier = SVC(kernel='rbf')
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of RBF SVM After PCA for n_components = 75: ",test_acc_pca)


# ### 6.5 PCA for n_components = 100

# In[45]:


### PCA

pca = PCA(n_components=100)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

print("Shape of Feature Test set Before PCA: ", X_train.shape)
print("Shape of Feature Test set After PCA for n_components = 100: ", X_train_pca.shape)


print("Shape of Feature Test set Before PCA: ", X_test.shape)
print("Shape of Feature Test set After PCA for n_components = 100: ", X_test_pca.shape)


# ### 6.5.1 Linear SVM

# In[46]:


###SVM - Linear

svmclassifier = SVC(kernel='linear')
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Linear SVM After PCA for n_components = 100: ",test_acc_pca)


# ### 6.5.2 Kernel SVM for polnomial degree 8, 5, 3, 2, 1

# In[47]:


svmclassifier = SVC(kernel='poly', degree=8)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 8 After PCA for n_components = 100: ",test_acc_pca)


# In[48]:


svmclassifier = SVC(kernel='poly', degree=5)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 5 After PCA for n_components = 100: ",test_acc_pca)


# In[49]:


svmclassifier = SVC(kernel='poly', degree=3)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 3 After PCA for n_components = 100: ",test_acc_pca)


# In[50]:


svmclassifier = SVC(kernel='poly', degree=2)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 2 After PCA for n_components = 100: ",test_acc_pca)


# In[51]:


svmclassifier = SVC(kernel='poly', degree=1)
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of Kernel SVM of degree 1 After PCA for n_components = 100: ",test_acc_pca)


# ### 6.5.3 RBF SVM

# In[52]:


svmclassifier = SVC(kernel='rbf')
svmclassifier.fit(X_train_pca, y_train)

y_pred_svmpca = svmclassifier.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_pred_svmpca)*100

print("Test accuracy of RBF SVM After PCA for n_components = 100: ",test_acc_pca)


# ### 7. SVM without PCA or LDA

# ### 7.1 Linear SVM without PCA or LDA

# In[53]:


### Without PCA or LDA

svmclassifier = SVC(kernel='linear')
svmclassifier.fit(X_train, y_train)

y_pred_svm = svmclassifier.predict(X_test)

test_acc = accuracy_score(y_test,y_pred_svm)*100

print("Test accuracy of Linear SVM without PCA and LDA: ",test_acc)


# ### 7.2 Kernel SVM without PCA or LDA (for 1, 2, 3, 5 and 8 degree polynomials) 

# In[54]:


svmclassifier = SVC(kernel='poly', degree=8)
svmclassifier.fit(X_train, y_train)

y_pred_svm = svmclassifier.predict(X_test)

test_acc = accuracy_score(y_test,y_pred_svm)*100

print("Test accuracy of Kernal SVM without PCA or LDA for 8 degree polynomial: ",test_acc)


# In[55]:


svmclassifier = SVC(kernel='poly', degree=5)
svmclassifier.fit(X_train, y_train)

y_pred_svm = svmclassifier.predict(X_test)

test_acc = accuracy_score(y_test,y_pred_svm)*100

print("Test accuracy of Kernal SVM without PCA or LDA for 5 degree polynmial: ",test_acc)


# In[56]:


svmclassifier = SVC(kernel='poly', degree=3)
svmclassifier.fit(X_train, y_train)

y_pred_svm = svmclassifier.predict(X_test)

test_acc = accuracy_score(y_test,y_pred_svm)*100

print("Test accuracy of Kernal SVM without PCA or LDA for 3 degree polynomial: ",test_acc)


# In[57]:


svmclassifier = SVC(kernel='poly', degree=2)
svmclassifier.fit(X_train, y_train)

y_pred_svm = svmclassifier.predict(X_test)

test_acc = accuracy_score(y_test,y_pred_svm)*100

print("Test accuracy of Kernal SVM without PCA or LDA for 2 degree polynomial: ",test_acc)


# In[58]:


svmclassifier = SVC(kernel='poly', degree=1)
svmclassifier.fit(X_train, y_train)

y_pred_svm = svmclassifier.predict(X_test)

test_acc = accuracy_score(y_test,y_pred_svm)*100

print("Test accuracy of Kernal SVM without PCA or LDA 1 polynomial: ",test_acc)


# ### 7.3 Radial Basis Function kernel SVM without PCA or LDA

# In[59]:


svmclassifier = SVC(kernel='rbf')
svmclassifier.fit(X_train, y_train)

y_pred_svm = svmclassifier.predict(X_test)

test_acc = accuracy_score(y_test,y_pred_svm)*100

print("Test accuracy of RBF SVM Without PCA or LDA : ",test_acc)


# In[ ]:




