import numpy as np
from numpy.random import uniform
import scipy as sp
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression as lr, Lasso as ls, Ridge as rg, ElasticNet as en, LogisticRegression as loglr
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.ensemble import RandomForestClassifier as rfc, GradientBoostingClassifier as gbc, VotingClassifier, BaggingClassifier, AdaBoostClassifier as abc
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler, LabelBinarizer
#from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import xgboost as xgb
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.model_selection import cross_val_score as cvs
from matplotlib import pyplot as plt
from datetime import datetime as dt
from sklearn.metrics import r2_score, mean_squared_log_error as msle
from sklearn.model_selection import KFold, train_test_split as tts
from catboost import CatBoostRegressor as cbr
from sklearn.tree import DecisionTreeRegressor as dtr, DecisionTreeClassifier as dtc
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer as pst
from mpl_toolkits.mplot3d import Axes3D as trid
from sklearn.decomposition import PCA, FactorAnalysis, KernelPCA
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Convolution2D, Flatten, MaxPooling2D, AveragePooling2D, Conv2D, LeakyReLU, BatchNormalization
from keras.utils import to_categorical
from keras.datasets import mnist
import keras as k
import h5py
from keras import regularizers
from keras.preprocessing.image import load_img, img_to_array
from statsmodels.tsa.seasonal import seasonal_decompose as sd
import cv2
import glob
mms = MinMaxScaler()
rs = RobustScaler()
ss = StandardScaler()
#imp = SimpleImputer(strategy = 'median')
lb = LabelBinarizer()
le = LabelEncoder()
pca = PCA()
kpca = KernelPCA(kernel = 'poly', random_state = 1)
fa = FactorAnalysis()
lda = LinearDiscriminantAnalysis()
tensorboard_callback = k.callbacks.TensorBoard(log_dir = 'logs/')



#downloading and loading dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#plotting some of the training examples
plt.subplot(221)
plt.imshow(X_train[0], cmap = plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap = plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[3], cmap = plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[4], cmap = plt.get_cmap('gray'))


#CNN
#changing input shape for convolution
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#rescaling
X_train = X_train/255
X_test = X_test/255

#converting classes to categorical dummies
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


def model_2():
    #instamtiate
    clf2 = Sequential()
    clf2.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform'))
    #clf2.add(LeakyReLU(alpha = 0.01))
    #clf2.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform'))
    clf2.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))
    clf2.add(BatchNormalization())
    clf2.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform'))
    #clf2.add(Conv2D(16, (3, 3), padding = 'same', kernel_initializer = 'he_uniform', activation = 'relu'))
    #clf2.add(Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_uniform', activation = 'relu'))
    #clf2.add(LeakyReLU(alpha = 0.01))
    clf2.add(MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'same'))
    clf2.add(Dropout(0.2))
    clf2.add(Flatten())
    clf2.add(Dense(120, activation = 'relu', kernel_initializer = 'he_uniform'))
    #clf2.add(Dropout(.2))
    #clf2.add(LeakyReLU(alpha = 0.01))
    #clf2.add(Dense(, activation = 'relu', kernel_initializer = 'he_uniform'))
    clf2.add(BatchNormalization())
    clf2.add(Dropout(.2))
    #clf2.add(LeakyReLU(alpha = 0.01))
    clf2.add(Dense(num_classes, activation = 'softmax'))
    clf2.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return clf2
 
model_2().summary()

temp_2 = model_2().fit(X_train, y_train, epochs = 15, validation_split = .2, shuffle = 'True', validation_data = (X_test, y_test))


#plot
plt.plot(temp_2.history['acc'])
plt.plot(temp_2.history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc = 'upper left')
plt.title('accuracy-CNN')
plt.show()

plt.plot(temp_2.history['loss'])
plt.plot(temp_2.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc = 'upper left')
plt.title('loss-CNN')
plt.show()



score = model_2().evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json = model_2().to_json()
with open("flask_deploy/model.json", "w") as json_file:
  json_file.write(model_json)

model_2().save_weights("flask_deploy/model.h5")  