import pickle
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC


def preprocess(X):
    #均值滤波
    fsize=5
    filter = np.ones(fsize) / fsize
    feature = X[:,[3]]
    feature_new = np.zeros((feature.shape[0],feature.shape[1], feature.shape[2]-fsize+1))
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            for pos in range(feature_new.shape[2]):
                feature_new[i,j,pos] = feature[i,j,pos:pos+fsize].dot(filter)
    #差分
    feature_new=np.diff(feature_new)
    feature_new=feature_new.reshape((feature_new.shape[0],feature_new.shape[1]*feature_new.shape[2]))
    return feature_new

def shallow():
    trainX, trainY, testX, testY = pickle.load(open('../data/data_clf.pkl', 'rb'))
    trainXX = preprocess(trainX)
    testXX = preprocess(testX)

    ids = np.arange(len(trainY))
    np.random.shuffle(ids)

    # clf = LinearSVC(dual=True)
    # for c in np.arange(-7,-5,0.05):
    #     clf=svm.SVC(kernel='linear',C=pow(10,c))
    #     clf.fit(trainXX[ids], trainY[ids])
    #     print(c, clf.score(testXX, testY))
    #     #c=1e-6.3 0.9570

    clf = svm.SVC(kernel='linear', C=pow(10, -6.3))
    clf.fit(trainXX[ids], trainY[ids])
    print(clf.score(testXX, testY))

    # c_range = np.logspace(-5, 15, 11, base=2)
    # gamma_range = np.logspace(-9, 3, 13, base=2)
    # param_grid = [{'kernel': ['rbf','linear'], 'C': c_range, 'gamma': gamma_range}]
    # grid = GridSearchCV(svm.SVC(), param_grid, cv=3, n_jobs=-1)
    # clf = grid.fit(trainXX[ids], trainY[ids])
    # print(clf.score(testXX, testY))

    # clf = RandomForestClassifier(n_estimators=300)
    # clf.fit(trainXX[ids], trainY[ids])
    # print(clf.score(testXX, testY))


if __name__=="__main__":
    shallow()
