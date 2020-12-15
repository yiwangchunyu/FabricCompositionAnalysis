import os
import random
import pickle
import lmdb
from sklearn import svm, metrics
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import pickle

TRAIN_CLF_DIR = 'data/train_clf'
TEST_CLF_DIR = 'data/test_clf'
TRAIN_REG_DIR = 'data/train_reg'
TEST_REG_DIR = 'data/test_reg'

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.lmdbData=LmdbData(data_dir)

    def __len__(self):
        return self.lmdbData.len()

    def __getitem__(self, idx):
        x,y=self.lmdbData.get(idx)
        x=x[:224,1]

        x = np.expand_dims(x, axis=0).astype(np.float32)
        return x,y

class LmdbData:
    def __init__(self,dir,size = 1):
        self.env = lmdb.open(dir, map_size=size*1024*1024) #size M
        self.next=0
        pass
    def len(self):
        txn = self.env.begin()
        len = int(txn.get('len'.encode()).decode())
        return len

    def get(self, id):
        txn = self.env.begin()
        x = pickle.loads(txn.get(('x_%d'%(id)).encode()))
        y = int(txn.get(('y_%d' % (id)).encode()).decode())
        return x,y
    def list(self):
        len=self.len()
        data=[]
        labels=[]
        for id in range(len):
            x,y=self.get(id)
            data.append(x)
            labels.append(y)
        return np.asarray(data), np.asarray(labels)

    def add(self,x,y):
        txn = self.env.begin(write=True)
        txn.put(('x_%d'%(self.next)).encode(), pickle.dumps(x))
        txn.put(('y_%d' % (self.next)).encode(), str(y).encode())
        txn.put('len'.encode(), str(self.next+1).encode())
        txn.commit()
        self.next+=1

    def put(self,k,v):
        txn = self.env.begin(write=True)
        txn.put(k, v)
        txn.commit()
    def finish(self):
        self.put('len'.encode(), str(self.next).encode())

def process_dir_num(dir, lmdb, y):
    for file in os.listdir(dir):
        if file.split('.')[-1]=='csv' and file.split('.')[-2].split('_')[-1] not in {'a','i','r'}:
            data=[]
            path=os.path.join(dir,file)
            print('prcessing file: %s' % path)
            with open(path,'r',encoding='gb18030', errors='ignore') as f:
                tag=False
                for line in f.readlines():
                    if tag==False and line[:10]=='Wavelength':
                        tag=True
                        continue
                    if tag:
                        row = list(map(float, line.split(',')))
                        data.append(row)
            lmdb.add(np.array(data),y)

def process_dir(dir, lmdb, y):
    for d in os.listdir(dir):
        path = os.path.join(dir,d)
        process_dir_num(path,lmdb,y)

def process_raw_clf(data_root='data'):
    train_root = os.path.join(data_root,'train')
    test_root = os.path.join(data_root, 'test')

    #分类任务，处理训练集数据
    lmdb_train_clf = LmdbData(TRAIN_CLF_DIR, size=30)
    train_cotton_dir='data/train/cotton'
    process_dir(train_cotton_dir,lmdb_train_clf,1)
    train_cotton_apandex_dir = 'data/train/cotton_spandex'
    process_dir(train_cotton_apandex_dir, lmdb_train_clf, 0)
    lmdb_train_clf.finish()
    # 分类任务，处理测试集数据
    lmdb_test_clf = LmdbData(TEST_CLF_DIR, size=30)
    test_cotton_dir = 'data/test/cotton'
    process_dir(test_cotton_dir, lmdb_test_clf, 1)
    test_cotton_apandex_dir = 'data/test/cotton_spandex'
    process_dir(test_cotton_apandex_dir, lmdb_test_clf, 0)
    lmdb_test_clf.finish()


def process_dir_num_reg(dir,X,Y,y):
    for file in os.listdir(dir):
        if file.split('.')[-1]=='csv' and file.split('.')[-2].split('_')[-1] not in {'a','i','r'}:
            data=[]
            path=os.path.join(dir,file)
            print('prcessing file: %s' % path)
            with open(path,'r',encoding='gb18030', errors='ignore') as f:
                tag=False
                for line in f.readlines():
                    if tag==False and line[:10]=='Wavelength':
                        tag=True
                        continue
                    if tag:
                        row = list(map(float, line.split(',')))
                        data.append(row[1])
            X.append(data)
            Y.append(y)

def process_dir_reg(dir, abundance, X, Y):
    for d in os.listdir(dir):
        y=abundance[d]
        path = os.path.join(dir,d)
        process_dir_num_reg(path,X,Y,y)

def process_raw_reg(data_root='data'):
    df=pd.read_excel(os.path.join(data_root,'labels.xlsx'),header=None)
    lst = df.values.tolist()
    abundance={}
    for row in lst:
        abundance[str(int(row[0]))]=row[2]
    train_cotton_apandex_dir = 'data/train/cotton_spandex'
    X=[]
    Y=[]
    process_dir_reg(train_cotton_apandex_dir, abundance, X, Y)
    train_data=[np.array(X),np.array(Y)]

    test_cotton_apandex_dir = 'data/test/cotton_spandex'
    X = []
    Y = []
    process_dir_reg(test_cotton_apandex_dir, abundance, X, Y)
    test_data = [np.array(X), np.array(Y)]

    pickle.dump(train_data,open('data/train_reg.pkl','wb'))
    pickle.dump(test_data, open('data/test_reg.pkl', 'wb'))

if __name__=="__main__":
    # process_raw_clf()
    process_raw_reg()

    pass