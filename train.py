import numpy as np
from sklearn import svm, neighbors

from dataset import LmdbData, TRAIN_CLF_DIR, TEST_CLF_DIR


def SVM():
    # Wavelength (nm),Absorbance (AU),Reference Signal (unitless),Sample Signal (unitless)
    feature_id=1
    lmdb_train_clf = LmdbData(TRAIN_CLF_DIR, size=30)
    data, labels = lmdb_train_clf.list()
    feat = data[:, :, feature_id]
    intensity = data[:, :, 3]
    # diff1= np.diff(feat)
    # diff2 = np.diff(diff1)
    # feat = np.concatenate((feat, diff1, diff2),axis=1)
    # feat = np.concatenate((feat, intensity), axis=1)

    lmdb_test_clf = LmdbData(TEST_CLF_DIR, size=30)
    data_test, labels_test = lmdb_test_clf.list()
    feat_test = data_test[:, :, feature_id]
    intensity = data_test[:, :, 3]
    # diff1 = np.diff(feat_test)
    # diff2 = np.diff(diff1)
    # feat_test = np.concatenate((feat_test, diff1, diff2), axis=1)
    # feat_test = np.concatenate((feat_test, intensity), axis=1)

    ids = np.arange(len(labels))
    np.random.shuffle(ids)
    X = feat[ids,:]
    Y = labels[ids]
    clf = svm.SVC()
    # clf = neighbors.KNeighborsClassifier(n_neighbors=2)
    print("SVC fitting...")
    clf.fit(X, Y)
    print('Train score:', clf.score(X, Y))
    print('Test score:', clf.score(feat_test, labels_test))

if __name__=="__main__":
    SVM()