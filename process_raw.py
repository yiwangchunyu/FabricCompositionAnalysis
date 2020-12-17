import os
import pickle
import pandas as pd
import numpy as np

raw_data_root='data/data.ncignore'
test_data = os.path.join(raw_data_root,'test')
train_data = os.path.join(raw_data_root,'train')
cotton='cotton'
cotton_spandex='cotton_spandex'
label_path=os.path.join(raw_data_root,'labels.xlsx')


def process_dir_clf(dir, X, Y, label):
    for d in os.listdir(dir):
        data = []
        for file in os.listdir(os.path.join(dir, d)):
            if file.split('.')[-1] == 'csv' and file.split('.')[-2].split('_')[-1] not in {'a', 'i', 'r'}:
                data_infile = []
                path = os.path.join(dir, d, file)
                print('prcessing file: %s' % path)
                with open(path, 'r', encoding='gb18030', errors='ignore') as f:
                    tag = False
                    for line in f.readlines():
                        if tag == False and line[:10] == 'Wavelength':
                            tag = True
                            continue
                        if tag:
                            row = list(map(float, line.split(',')))
                            data_infile.append(row)
                data.append(np.asarray(data_infile).T)
                onesample = np.asarray(data_infile).T
                # read reflectance
                rfile=file.split('.')[0]+'_r.csv'
                rpath=os.path.join(dir, d, rfile)
                data_infile = []
                with open(rpath, 'r', encoding='gb18030', errors='ignore') as f:
                    tag = False
                    for line in f.readlines():
                        if tag == False and line[:10] == 'Wavelength':
                            tag = True
                            continue
                        if tag:
                            row = list(map(float, line.split(',')))
                            data_infile.append(row)
                onesample[2]=np.array(data_infile)[:,1]
                Y.append(label)
                X.append(onesample)
        # X.append(np.asarray(data)[:20])
        # Y.append(label)

def process_dir_reg(dir, X, Y, label):
    for d in os.listdir(dir):
        y=label[d]
        data = []
        for file in os.listdir(os.path.join(dir, d)):
            if file.split('.')[-1] == 'csv' and file.split('.')[-2].split('_')[-1] not in {'a', 'i', 'r'}:
                data_infile = []
                path = os.path.join(dir, d, file)
                print('prcessing file: %s' % path)
                with open(path, 'r', encoding='gb18030', errors='ignore') as f:
                    tag = False
                    for line in f.readlines():
                        if tag == False and line[:10] == 'Wavelength':
                            tag = True
                            continue
                        if tag:
                            row = list(map(float, line.split(',')))
                            data_infile.append(row)
                data.append(np.asarray(data_infile).T)
                onesample = np.asarray(data_infile).T
                # read reflectance
                rfile = file.split('.')[0] + '_r.csv'
                rpath = os.path.join(dir, d, rfile)
                data_infile = []
                with open(rpath, 'r', encoding='gb18030', errors='ignore') as f:
                    tag = False
                    for line in f.readlines():
                        if tag == False and line[:10] == 'Wavelength':
                            tag = True
                            continue
                        if tag:
                            row = list(map(float, line.split(',')))
                            data_infile.append(row)
                onesample[2] = np.array(data_infile)[:, 1]
                Y.append(y)
                X.append(onesample)

def gen_data_clf():
    trainX=[]
    trainY=[]
    testX=[]
    testY=[]
    #处理train
    cotton_dir=os.path.join(train_data,cotton)
    process_dir_clf(cotton_dir,trainX,trainY,1)
    cotton_spandex_dir = os.path.join(train_data, cotton_spandex)
    process_dir_clf(cotton_spandex_dir, trainX, trainY, 0)

    # 处理test
    cotton_dir = os.path.join(test_data, cotton)
    process_dir_clf(cotton_dir, testX, testY, 1)
    cotton_spandex_dir = os.path.join(test_data, cotton_spandex)
    process_dir_clf(cotton_spandex_dir, testX, testY, 0)

    trainX=np.asarray(trainX)
    trainY = np.asarray(trainY)
    testX = np.asarray(testX)
    testY = np.asarray(testY)
    pickle.dump([trainX, trainY, testX, testY], open('data/data_clf.pkl', 'wb'))

def gen_data_reg():
    df = pd.read_excel(label_path, header=None)
    lst = df.values.tolist()
    abundance = {}
    for row in lst:
        abundance[str(int(row[0]))] = row[2]
    trainX=[]
    trainY=[]
    testX=[]
    testY=[]
    #处理train
    cotton_spandex_dir = os.path.join(train_data, cotton_spandex)
    process_dir_reg(cotton_spandex_dir, trainX, trainY, abundance)

    # 处理test
    cotton_spandex_dir = os.path.join(test_data, cotton_spandex)
    process_dir_reg(cotton_spandex_dir, testX, testY, abundance)

    trainX=np.asarray(trainX)
    trainY = np.asarray(trainY)
    testX = np.asarray(testX)
    testY = np.asarray(testY)
    pickle.dump([trainX, trainY, testX, testY], open('data/data_reg.pkl', 'wb'))

def process_dir_joint(dir, X, Y, label, is_cotton):
    for d in os.listdir(dir):
        if is_cotton==1:
            y=0
        else:
            y=label[d]
        data = []
        for file in os.listdir(os.path.join(dir, d)):
            if file.split('.')[-1] == 'csv' and file.split('.')[-2].split('_')[-1] not in {'a', 'i', 'r'}:
                data_infile = []
                path = os.path.join(dir, d, file)
                print('prcessing file: %s' % path)
                with open(path, 'r', encoding='gb18030', errors='ignore') as f:
                    tag = False
                    for line in f.readlines():
                        if tag == False and line[:10] == 'Wavelength':
                            tag = True
                            continue
                        if tag:
                            row = list(map(float, line.split(',')))
                            data_infile.append(row)
                data.append(np.asarray(data_infile).T)
                onesample = np.asarray(data_infile).T
                # read reflectance
                rfile = file.split('.')[0] + '_r.csv'
                rpath = os.path.join(dir, d, rfile)
                data_infile = []
                with open(rpath, 'r', encoding='gb18030', errors='ignore') as f:
                    tag = False
                    for line in f.readlines():
                        if tag == False and line[:10] == 'Wavelength':
                            tag = True
                            continue
                        if tag:
                            row = list(map(float, line.split(',')))
                            data_infile.append(row)
                onesample[2] = np.array(data_infile)[:, 1]
                Y.append([y,is_cotton])
                X.append(onesample)

def gen_data_joint():
    df = pd.read_excel(label_path, header=None)
    lst = df.values.tolist()
    abundance = {}
    for row in lst:
        abundance[str(int(row[0]))] = row[2]
    trainX=[]
    trainY=[]
    testX=[]
    testY=[]
    #处理train
    cotton_spandex_dir = os.path.join(train_data, cotton_spandex)
    process_dir_joint(cotton_spandex_dir, trainX, trainY, abundance,0)
    cotton_dir = os.path.join(train_data, cotton)
    process_dir_joint(cotton_dir, trainX, trainY, abundance, 1)

    # 处理test
    cotton_spandex_dir = os.path.join(test_data, cotton_spandex)
    process_dir_joint(cotton_spandex_dir, testX, testY, abundance,0)
    cotton_dir = os.path.join(test_data, cotton)
    process_dir_joint(cotton_dir, testX, testY, abundance,1)

    trainX=np.asarray(trainX)
    trainY = np.asarray(trainY)
    testX = np.asarray(testX)
    testY = np.asarray(testY)
    pickle.dump([trainX, trainY, testX, testY], open('data/data_joint.pkl', 'wb'))

if __name__=='__main__':
    # gen_data_clf()
    # gen_data_reg()
    gen_data_joint()