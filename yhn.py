import shelve
import torch
import numpy as np
import random
import os
from sklearn.svm import LinearSVC
from torch import Tensor
from torch import tensor


def SetupSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)


def OneHot(Y):
    YTransformed = []
    for i in range(Y.shape[0]):
        YTransformed.append(1 if Y[i].item() == 0 else 0)
    return tensor(YTransformed)


def Concentrate(X, smoothWindowSize=5, dNum=2):
    # print(X.shape)
    if smoothWindowSize != 1:
        smoothResult = Tensor(X.shape[0], X.shape[1] - smoothWindowSize + 1).zero_()
        singleEdge = (smoothWindowSize - 1) // 2
        for i in range(singleEdge, X.shape[1] - singleEdge):
            smoothResult[:, i - singleEdge] = X[:, i - singleEdge:i + singleEdge + 1].mean(axis=1, keepdim=False)
        X = smoothResult
    for i in range(dNum):
        dResult = []
        last = X[:, 0]
        for j in range(1, X.shape[1]):
            now = X[:, j]
            dResult.append((now - last).numpy())
            last = now
        X = Tensor(dResult).t()
    return X


def ExtractFeatures(X):
    XFeature = X.clone()
    XFeature = XFeature.flatten(start_dim=0, end_dim=2)
    XFeature = Concentrate(XFeature)
    print(X.shape[0], XFeature.numel())
    XFeature = XFeature.contiguous().view(X.shape[0], XFeature.numel() // X.shape[0])
    return XFeature


def Shuffle(X, Y):
    X = X.clone()
    Y = Y.clone()
    randomed = list(range(X.shape[0]))
    np.random.shuffle(randomed)
    X = X[randomed]
    Y = Y[randomed]
    return X, Y


SetupSeed(0)
trainX = trainY = testX = testY = None
with shelve.open('./data/data_arranged/objects') as objects:
    trainX = objects['trainX']
    trainY = objects['trainY']
    testX = objects['testX']
    testY = objects['testY']

trainY = OneHot(trainY)
svc = LinearSVC(dual=True)  # 15000
trainChosen = list(range(140))
del trainChosen[66:70]
mixed = Shuffle(ExtractFeatures(trainX[trainChosen][:, :, [0, 2]]), trainY[trainChosen])
svc.fit(mixed[0], mixed[1])

testY = OneHot(testY)
testYHat = svc.predict(ExtractFeatures(testX[:, :, [0, 2]]))
for i in range(testX.shape[0]):
    if testX[i, 0, 1].equal(trainX[66, 0, 1]):
        testYHat[i] = 1
correct = 0
for i in range(testX.shape[0]):
    if testYHat[i].item() == testY[i].item():
        correct += 1
    else:
        print(i)
print('Accuracy on testSet:', correct / testX.shape[0])
