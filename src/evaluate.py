# 數字轉換分類結果
ind = np.argmax(y_probs_train,1)
y_prdict = data_fin.Type.astype('category')
y_prdict = y_prdict.dtypes.categories
y_prdict = y_prdict[ind]
y_prdict = pd.Series(y_prdict)

# 訓練與驗證數據，訓練準確度
accuracy = accuracy_score(y_all_train, y_prdict)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# 畫出confusion matrix
cm = confusion_matrix(y_all_train,y_prdict)
ax = sns.heatmap(cm,annot=True,fmt="d")
ax.set(xlabel='Predicted', ylabel='Actual', title='90% Training Data')
np.shape(y_all_train)

figure = ax.get_figure()
output_file = rootpath+'Result/Plot/train_confusion_matrix.jpg'
figure.savefig(output_file)

# 數字轉換分類結果
ind = np.argmax(y_probs_test,1)
y_prdict_test = data_fin.Type.astype('category')
y_prdict_test = y_prdict_test.dtypes.categories
y_prdict_test = y_prdict_test[ind]
y_prdict_test = pd.Series(y_prdict_test)

# Test data，預測準確度
# evaluate predictions
accuracy = accuracy_score(y_all_test, y_prdict_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# 畫出confusion matrix
cm = confusion_matrix(y_all_test,y_prdict_test)
ax = sns.heatmap(cm,annot=True,fmt="d")
ax.set(xlabel='Predicted', ylabel='Actual', title='10% Testing Data')
np.shape(y_all_test)

figure = ax.get_figure()
output_file = rootpath+'Result/Plot/test_confusion_matrix.jpg'
figure.savefig(output_file)

# 數字轉換分類結果
ind = np.argmax(y_probs_all,1)
y_prdict_all = data_fin.Type.astype('category')
y_prdict_all = y_prdict_all.dtypes.categories
y_prdict_all = y_prdict_all[ind]
y_prdict_all = pd.Series(y_prdict_all)

ind = np.argmax(y_dummy_all,1)
y_all = data_fin.Type.astype('category')
y_all = y_all.dtypes.categories
y_all = y_all[ind]
y_all = pd.Series(y_all)

# Test data，預測準確度
# evaluate predictions
accuracy = accuracy_score(y_all, y_prdict_all)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# 畫出confusion matrix
cm = confusion_matrix(y_all,y_prdict_all)
ax = sns.heatmap(cm,annot=True,fmt="d")
ax.set(xlabel='Predicted', ylabel='Actual', title='90% Training and 10% Testing Data')
np.shape(y_all)

figure = ax.get_figure()
output_file = rootpath+'Result/Plot/all_confusion_matrix.jpg'
figure.savefig(output_file)