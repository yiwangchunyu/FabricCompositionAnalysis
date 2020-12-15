import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import LmdbData, TRAIN_CLF_DIR

color=['r','b']
# Wavelength (nm),Absorbance (AU),Reference Signal (unitless),Sample Signal (unitless)
feature_name=['Absorbance','Reference','Intensity']
feature_id=3
lmdb_train_clf = LmdbData(TRAIN_CLF_DIR, size=30)
data, labels = lmdb_train_clf.list()

plt.figure(figsize=(30,15))
for feature_id in tqdm(range(1,4)):
    plt.subplot(3, 3, feature_id)
    for i in range(len(labels)):
        plt.plot(data[i,:,0],data[i,:,feature_id],color=color[labels[i]], alpha=0.3)
    plt.title(feature_name[feature_id-1])

    ids1 = np.where(labels == 1)[0]
    plt.subplot(3, 3, 3+feature_id)
    for i in ids1:
        plt.plot(data[i, :, 0], data[i, :, feature_id], color=color[labels[i]], alpha=0.3)
    ids0 = np.where(labels == 0)[0]
    plt.subplot(3, 3, 6 + feature_id)
    for i in ids0:
        plt.plot(data[i, :, 0], data[i, :, feature_id], color=color[labels[i]], alpha=0.3)

plt.show()

data_diff=np.zeros((data.shape[0],data.shape[1],3))
data_diff[:,:,0]=data[:,:,1]
plt.figure(figsize=(30,15))