from google.colab import drive

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Reshape, Dropout
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

from keras import backend as K
import cv2

# Google Drive授權，可直接讀取Drive內檔案
drive.mount('/content/gdrive')

rootpath = '/content/gdrive/My Drive/Colab Notebooks/'

# Create a folder to store results

if not os.path.exists(rootpath+'Result'):
  os.mkdir(rootpath+'Result')
  print('Create: ',rootpath+'Result')

if not os.path.exists(rootpath+'Result/Bad_Curves_TrainingData'):
  os.mkdir(rootpath+'Result/Bad_Curves_TrainingData')
  print('Create: ',rootpath+'Result/Bad_Curves_TrainingData')

if not os.path.exists(rootpath+'Result/Model'):
  os.mkdir(rootpath+'Result/Model')
  print('Create: ',rootpath+'Result/Model')

if not os.path.exists(rootpath+'Result/Plot'):
  os.mkdir(rootpath+'Result/Plot')
  print('Create: ',rootpath+'Result/Plot')

if not os.path.exists(rootpath+'Result/Test_Prediction'):
  os.mkdir(rootpath+'Result/Test_Prediction')
  print('Create: ',rootpath+'Result/Test_Prediction')

if not os.path.exists(rootpath+'Result/Test_Wrong_Cat'):
  os.mkdir(rootpath+'Result/Test_Wrong_Cat')
  print('Create: ',rootpath+'Result/Test_Wrong_Cat')

if not os.path.exists(rootpath+'Result/VisualizationHeatMap_TrainTest'):
  os.mkdir(rootpath+'Result/VisualizationHeatMap_TrainTest')
  print('Create: ',rootpath+'Result/VisualizationHeatMap_TrainTest')

if not os.path.exists(rootpath+'Result/VisualizationHeatMap_TrueTest'):
  os.mkdir(rootpath+'Result/VisualizationHeatMap_TrueTest')
  print('Create: ',rootpath+'Result/VisualizationHeatMap_TrueTest')

# 進入google drive資料夾
os.chdir(rootpath)

# 秀出當前路徑內檔案，確認是否在正確的路徑下
os.listdir()

# 初賽訓練數據資料夾內子資料夾
ClassFolder = os.listdir(rootpath+'thubigdata2019training-230/大數據競賽初賽資料(230測試數據)')
print(ClassFolder)

# 利用迴圈將所有資料夾內檔案整合成一個dataframe
init_cnt = 0

for i in range(0,len(ClassFolder)):
  # 進入不同子資料夾以讀取檔案
  os.chdir(rootpath+'thubigdata2019training-230/大數據競賽初賽資料(230測試數據)/'+ClassFolder[i])
  FileName_tmp = os.listdir()

  for j in range(0,len(FileName_tmp)):
    data_tmp = pd.read_csv(FileName_tmp[j], sep='\\t', engine='python')
    data_tmp = data_tmp.transpose()
    data_tmp = data_tmp.drop([0],axis=1) #丟掉溫度單位(Deg.F)
    data_tmp = data_tmp.astype(float) #將string轉換成float
    data_tmp.insert(0,'Type',FileName_tmp[j][0:3])
    data_tmp.insert(0,'FileName', FileName_tmp[j])
    if init_cnt == 0:
      data = data_tmp
      init_cnt += 1
    else:
      data = pd.concat([data, data_tmp], ignore_index=True, sort=False)

data = data.reset_index()
data = data.rename(columns={"index":"Col"})

print(data.head())
print(data.shape)

m, n = data.shape
x_tmp1 = data.iloc[:,3:n].values.copy()
x_tmp2 = data.iloc[:,3:n].values.copy()
data_diff = data.copy()

for i in range(0,m):
  f = x_tmp1[i,0:n]

  ind = np.isnan(f)
  f[np.isnan(f)] = f[f.shape[0]-ind.sum()-1]

  fg = np.gradient(f)

  x_tmp1[i,0:n] = f
  x_tmp2[i,0:n] = fg

data.iloc[:,3:n] = x_tmp1
data_diff.iloc[:,3:n] = x_tmp2

# 對微分數值設閾值，篩選不穩定之時間溫度曲線
ind_bad = []
m, n = data_diff.shape

for i in range(0,m):
  tmp = data_diff.iloc[i,3:n].values.copy()

  if abs(tmp.min())>15 or abs(tmp.max())>15:
    ind_bad = ind_bad + [i]

# 將品質壞掉的時間溫度曲線拿掉
data_fin = data.drop(data.index[ind_bad])
data_diff_fin = data_diff.drop(data_diff.index[ind_bad])

print('Original shape:',data.shape)
print('Final shape:',data_fin.shape)

# 隨機分配數據為90%訓練和10%測試
fun_1 = lambda x: x.sample(frac=0.9, replace=False, random_state=1)

data_fin_Train = data_diff_fin.groupby('Type', group_keys=False).apply(fun_1)
data_fin_Test = data_diff_fin.drop(data_fin_Train.index)

print('Train shape:',data_fin_Train.shape)
print('Test shape:',data_fin_Test.shape)

# 檢查在Train和Test資料分布比例是否一致
type_counts_Train = data_fin_Train.groupby('Type', group_keys=False).count()
type_counts_Train = type_counts_Train['Col']
type_counts_Train = type_counts_Train/type_counts_Train.sum()*100
type_counts_Train.index.name = "Type (%)"

type_counts_Test = data_fin_Test.groupby('Type', group_keys=False).count()
type_counts_Test = type_counts_Test['Col']
type_counts_Test = type_counts_Test/type_counts_Test.sum()*100
type_counts_Test.index.name = "Type (%)"

print('Train: ',type_counts_Train)
print('Test: ',type_counts_Test)

# 設置x和y
x_all_train = data_fin_Train.drop(['Col','FileName','Type'],axis=1)
x_all_test = data_fin_Test.drop(['Col','FileName','Type'],axis=1)

# 將y做成one-hot encoding
# 要注意不同模型訓練的時候丟入的y有可能是原本0和1(y_train)，或是經過one-hot encoding的(y_dummy)
y_all_train = data_fin_Train['Type']
y_all_dummy_train = pd.get_dummies(y_all_train)
y_all_test = data_fin_Test['Type']
y_all_dummy_test = pd.get_dummies(y_all_test)


# 檢視設置結果
print('train:')
print('x:',x_all_train.shape)
print(x_all_train.head(3))
print('y:',y_all_dummy_train.shape)
print(y_all_dummy_train.head(3))
print()
print('test:')
print('x:',x_all_test.shape)
print(x_all_test.head(3))
print('y:',y_all_dummy_test.shape)
print(y_all_dummy_test.head(3))

# 在您的代码中添加以下代码：
import keras
import tensorflow

print("Keras version:", keras.__version__)
print("TensorFlow version:", tensorflow.__version__)

# 開始進行cross-validation

n_folds = 5
cv_scores, model_history, train_history = list(), list(), list()

for _ in range(n_folds):
  # split data
  X_train, X_val, y_train, y_val = \
      train_test_split(x_all_train, y_all_dummy_train, test_size=1/n_folds, \
      random_state = np.random.randint(1,1000, 1)[0], stratify=y_all_dummy_train)

  # evaluate model
  model, val_acc, history = evaluate_model(X_train, y_train, 'model_CV.weights.h5', X_val, y_val)
  print('>%.3f' % val_acc)

  cv_scores.append(val_acc)
  model_history.append(model)
  train_history.append(history)


  print("train_history length:", len(train_history))

#交叉驗證準確度的平均(標準差)
print(cv_scores)
print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))

# 準確度
show_train_history(train_history,'accuracy','val_accuracy')

# Loss
show_train_history(train_history,'loss','val_loss')

# 訓練90%訓練數據，建構模型
model, val_acc, history = evaluate_model(x_all_train, y_all_dummy_train, 'model_CV_fin.weights.h5')
print('>%.3f' % val_acc)

# 預測訓練模型並檢視結果
y_probs_train = model.predict(x_all_train) #利用訓練後模型預測train資料的結果
print(y_probs_train.shape) #矩陣大小
print(y_probs_train[0:4,:]) #只顯示前五列

y_probs_test = model.predict(x_all_test)
print(y_probs_test.shape) #矩陣大小
print(y_probs_test[0:4,:]) #只顯示前五列

model.summary()

print("x_all_test type:", type(x_all_test))
print("x_all_test shape:", x_all_test.shape)
print(x_all_test.head())


print("rootpath:", rootpath)


print("data_fin type:", type(data_fin))
print("data_fin columns:", data_fin.columns)
print(data_fin.head())
print("data_fin.Type dtype:", data_fin.Type.dtype)
# 如果 data_fin.Type 應轉為 categorical，可以檢查
data_fin["Type"] = data_fin["Type"].astype("category")
print("data_fin.Type categories:", data_fin["Type"].cat.categories)


print("model type:", type(model))
model.summary()
print("model.input_shape:", model.input_shape)


import numpy as np
# 假設單筆資料的原始 shape 應該是 (449,)
sample = np.array(x_all_test.iloc[0, :], dtype=np.float32)
print("單筆資料原始 shape:", sample.shape)
# 如果模型第一層是 Reshape((449, 1))，你應該傳入 (1, 449)
sample = np.expand_dims(sample, axis=0)
print("傳入模型的資料 shape:", sample.shape)


# 將訓練與測試數據結合
x_all = np.append(x_all_train,x_all_test,axis=0)
y_dummy_all = np.append(y_all_dummy_train,y_all_dummy_test,axis=0)

# 訓練全部的初賽訓練數據，建構最後模型
model, val_acc, history = evaluate_model(x_all, y_dummy_all, 'model_fin.weights.h5')
print('>%.3f' % val_acc)

y_probs_all = model.predict(x_all)
print(y_probs_all.shape) #矩陣大小
print(y_probs_all[0:4,:]) #只顯示前五列

# 進入初賽訓練數據資料夾
os.chdir(rootpath+'thubigdata2019exam-722')

# 秀出當前路徑內檔案，確認是否在正確的路徑下
files = [f for f in os.listdir('.') if os.path.isfile(f)]

print(files)
print('Number of files:',len(files))

# 利用迴圈將所有資料夾內檔案整合成一個dataframe
init_cnt = 0
os.chdir(rootpath+'thubigdata2019exam-722')

for i in files:
  data_tmp = pd.read_csv(i, sep='\\t', engine='python', index_col=False)
  data_tmp = data_tmp.transpose()
  data_tmp = data_tmp.drop([0],axis=1) #丟掉溫度單位(Deg.F)
  data_tmp = data_tmp.astype(float) #將string轉換成float

  data_tmp.insert(0,'Type',np.nan)
  data_tmp.insert(0,'FileName',i)

  if init_cnt == 0:
    data_test = data_tmp
    init_cnt += 1
  else:
    # Use pd.concat instead of append
    data_test = pd.concat([data_test, data_tmp], ignore_index=True, sort=False)

data_test = data_test.reset_index()
data_test = data_test.rename(columns={"index":"Col"})

# 加長時間長度和訓練一樣
for i in range(data_test.shape[1]-3,data.shape[1]-2):
  data_test[i] = np.nan

print(data_test.head())
print(data_test.shape)

m, n = data_test.shape
x_tmp1 = data_test.iloc[:,3:n].values.copy()
x_tmp2 = data_test.iloc[:,3:n].values.copy()
data_test_diff = data_test.copy()

for i in range(0,m):
  f = x_tmp1[i,0:n]

  ind = np.isnan(f)
  f[np.isnan(f)] = f[f.shape[0]-ind.sum()-1]

  fg = np.gradient(f)

  x_tmp1[i,0:n] = f
  x_tmp2[i,0:n] = fg

data_test.iloc[:,3:n] = x_tmp1
data_test_diff.iloc[:,3:n] = x_tmp2

# 對微分數值設閾值，篩選不穩定之時間溫度曲線
ind_bad_test = []
m, n = data_test_diff.shape

for i in range(0,m):
  tmp = data_test_diff.iloc[i,3:n].values.copy()

  if abs(tmp.min())>15 or abs(tmp.max())>15:
    ind_bad_test = ind_bad_test + [i]

# 設置x和y
x_true_test = data_test_diff.drop(['Col','FileName','Type'],axis=1)

# 將y做成one-hot encoding
# 要注意不同模型訓練的時候丟入的y有可能是原本0和1(y_train)，或是經過one-hot encoding的(y_dummy)
y_true_test = data_test_diff['Type']
y_dummy_true_test = pd.get_dummies(y_true_test)

# 檢視設置結果
print('true test:')
print('x:',x_true_test.shape)
print(x_true_test.head(3))
print('y:',y_dummy_true_test.shape)
print(y_dummy_true_test.head(3))

# 進入儲存模型資料夾
os.chdir(rootpath+'Result/Model')
model.load_weights('model_fin.h5') #讀取模型

# 預測測試數據結果
y_probs_true_test = model.predict(x_true_test)
print(y_probs_true_test.shape) #矩陣大小
print(y_probs_true_test[0:4,:]) #只顯示前五列

# 數字轉換分類結果
ind = np.argmax(y_probs_true_test,1)
y_prdict_true_test = data_fin.Type.astype('category')
y_prdict_true_test = y_prdict_true_test.dtypes.categories
y_prdict_true_test = y_prdict_true_test[ind]
y_prdict_true_test = pd.Series(y_prdict_true_test)

# 將預測結果填入
data_test['Type'] = y_prdict_true_test

# 檔名重複的index
du_index = data_test['FileName'].duplicated()

# 每個檔案的開頭與結尾index
tmp_index = data_test.index[~du_index].tolist()
file_index = []
file_name = []

for i in range(0,len(tmp_index)-1):
  file_index = file_index+[[tmp_index[i],tmp_index[i+1]]]
  file_name = file_name+[data_test['FileName'][tmp_index[i]]]

file_index = file_index+[[tmp_index[i+1],data_test.shape[0]]]
file_name = file_name+[data_test['FileName'][tmp_index[i+1]]]

print(file_index)
print(file_name)

# 統計每個檔案分類的結果
file_name_int = []
file_cat = []
file_with_diff_cat = []

for i in range(0,len(file_name)):

  # 轉換檔名成整數
  (name,ext) = os.path.splitext(file_name[i])
  file_name_int = file_name_int+[int(name)]

  # 統計分類結果
  tmp = data_test['Type'][list(range(file_index[i][0],file_index[i][1]))]
  tmp = tmp.astype('category')
  cat_tmp = tmp.value_counts()
  cat_tmp = cat_tmp.index[0]
  file_cat = file_cat+[cat_tmp]

  if len(tmp.dtypes.categories)!=1:
    file_with_diff_cat = file_with_diff_cat+[i]

# 輸出比賽格式結果

# 將完整的測驗數據DataFrame輸出
os.chdir(rootpath+'Result/Test_Prediction')
data_test.to_excel('TestResult_all.xlsx', index = None, header=True)

# 檔名排序
files_sort_ind = sorted(range(len(file_name_int)), key=lambda k: file_name_int[k])

# 結果DataFrame
data_test_res = pd.DataFrame(list(range(1,len(files_sort_ind)+1)))
data_test_res[1] = [file_cat[i] for i in files_sort_ind]
data_test_res.columns = ['測驗數據資料代號','分類結果']

# 輸出成excel的xlsx格式
os.chdir(rootpath+'Result/Test_Prediction')
data_test_res.to_excel('2_108059_TestResult.xlsx', index = None, header=True)

