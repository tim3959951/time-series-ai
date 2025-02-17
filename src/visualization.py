# 畫出train前5個微分的時間溫度曲線
subplots_adjust(hspace=0.000)
number_of_subplots=5

for i,v in enumerate(range(number_of_subplots)):
    v = v+1
    ax1 = subplot(number_of_subplots,1,v)
    ax1.plot(data_diff.iloc[i,3:n])

    if i == 0:
      plt.title('Derivative Curves')
    elif i == round(number_of_subplots/2):
      plt.ylabel('Temperature Gradient')
    elif i == number_of_subplots-1:
      plt.xlabel('Time')

plt.show()

figure=ax1.get_figure()
output_file = rootpath+'Result/Plot/derivative_curves.jpg'
figure.savefig(output_file)

# 繪製不穩定之時間溫度曲線
m, n = data.shape

for i in ind_bad:
  str_1 = str(i).zfill(3);

  plt.plot(data.iloc[i,3:n].values)
  plt.title('number_'+str_1)
  output_file = rootpath+'Result/Bad_Curves_TrainingData/'+str_1+'.jpg'
  plt.savefig(output_file)

  plt.figure
  plt.show()

# 畫出前5個壞掉的時間溫度曲線
subplots_adjust(hspace=0.000)
number_of_subplots=5

for i,v in enumerate(range(number_of_subplots)):
    v = v+1
    ax1 = subplot(number_of_subplots,1,v)
    ax1.plot(data.iloc[ind_bad[i],3:data.shape[1]])

    if i == 0:
      plt.title('Broken Curves')
    elif i == round(number_of_subplots/2):
      plt.ylabel('Temperature')
    elif i == number_of_subplots-1:
      plt.xlabel('Time')

plt.show()

figure=ax1.get_figure()
output_file = rootpath+'Result/Plot/broken_curves.jpg'
figure.savefig(output_file)

# 畫出前5個壞掉的時間溫度曲線
subplots_adjust(hspace=0.000)
number_of_subplots=5

for i,v in enumerate(range(number_of_subplots)):
    v = v+1
    ax1 = subplot(number_of_subplots,1,v)
    ax1.plot(data_diff.iloc[ind_bad[i],3:data.shape[1]])

    if i == 0:
      plt.title('Broken Derivative Curves')
    elif i == round(number_of_subplots/2):
      plt.ylabel('Temperature Gradient')
    elif i == number_of_subplots-1:
      plt.xlabel('Time')

plt.show()

figure=ax1.get_figure()
output_file = rootpath+'Result/Plot/broken_derivative_curves.jpg'
figure.savefig(output_file)

def show_train_history(train_history,train,validation):
  subplots_adjust(hspace=0.000)
  number_of_subplots=len(train_history)

  for i,v in enumerate(range(number_of_subplots)):
    v = v+1
    ax1 = subplot(number_of_subplots,1,v)
    ax1.plot(train_history[i].history[train])
    ax1.plot(train_history[i].history[validation])

    if i == 0:
      plt.title('Train History')
    elif i == round(number_of_subplots/2):
      plt.ylabel(train)
    elif i == number_of_subplots-1:
      plt.xlabel('Epoch')
      plt.legend(['train', 'validation'], loc='upper right')

  plt.show()

  figure=ax1.get_figure()
  output_file = rootpath+'Result/Plot/cv_training_history_'+train+'.jpg'
  figure.savefig(output_file)

# 畫出test前5個微分的時間溫度曲線
subplots_adjust(hspace=0.000)
number_of_subplots=5

for i,v in enumerate(range(number_of_subplots)):
    v = v+1
    ax1 = subplot(number_of_subplots,1,v)
    ax1.plot(data_test_diff.iloc[i,3:n])

    if i == 0:
      plt.title('Derivative Curves')
    elif i == round(number_of_subplots/2):
      plt.ylabel('Temperature Gradient')
    elif i == number_of_subplots-1:
      plt.xlabel('Time')

plt.show()

figure=ax1.get_figure()
output_file = rootpath+'Result/Plot/derivative_curves_test.jpg'
figure.savefig(output_file)

# 畫出壞掉的時間溫度曲線
subplots_adjust(hspace=0.000)
number_of_subplots=len(ind_bad_test)

for i,v in enumerate(range(number_of_subplots)):
    v = v+1
    ax1 = subplot(number_of_subplots,1,v)
    ax1.plot(data_test.iloc[ind_bad_test[i],3:data_test.shape[1]])

    if i == 0:
      plt.title('Broken Curves')
    elif i == round(number_of_subplots/2)-1:
      plt.ylabel('Temperature')
    elif i == number_of_subplots-1:
      plt.xlabel('Time')

plt.show()

figure=ax1.get_figure()
output_file = rootpath+'Result/Plot/broken_curves_true_test.jpg'
figure.savefig(output_file)

# 將所有測試數據都進行可視化分析，並輸出成圖檔儲存
cnt = 0
for i in range(0,x_true_test.shape[0]):
  str_1 = str(i).zfill(3);
  print(str_1)
  output_file = rootpath+'Result/VisualizationHeatMap_TrueTest/test_' + str_1 + '.jpg'

  print( "Test data number: ", i)
  test_num = i

  x = x_true_test.iloc[test_num,:]
  x = np.expand_dims(x, axis=0)

  # 取得曲線分類類別
  preds = model.predict(x)
  # pred_class = np.argmax(preds[0])
  L = np.argsort(-preds, axis=1)
  L = L.squeeze()
  # 將preds排序，L為其排序過的index，第1個為最大L[0]，第二個為第二大L[1]，依此類推
  pred_class = L[0]

  # 取得曲線分類名稱
  y_class_name = data_fin.Type.astype('category')
  y_class_name = y_class_name.dtypes.categories
  y_class_name = y_class_name[pred_class]

  # 預測分類的輸出向量
  pred_output = model.output[:, pred_class]
  #print('pred_output.shape = ',pred_output.shape)

  # 最後一層 convolution layer 輸出的 feature map
  if cnt == 0:
    print(model.summary())
    print('Selected layer: ', model.layers[5].name)
    cnt = cnt+1

  last_conv_layer = model.get_layer(model.layers[5].name)

  # 求得分類的神經元對於最後一層 convolution layer 的梯度
  grads = K.gradients(pred_output, last_conv_layer.output)[0]

  # 求得針對每個 feature map 的梯度加總
  pooled_grads = K.sum(grads, axis=(0, 1))

  # K.function() 讓我們可以藉由輸入影像至 `model.input` 得到 `pooled_grads` 與
  # `last_conv_layer[0]` 的輸出值，像似在 Tensorflow 中定義計算圖後使用 feed_dict
  # 的方式。
  iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

  # 傳入影像矩陣 x，並得到分類對 feature map 的梯度與最後一層 convolution layer 的
  # feature map
  # Sam: pooled_grads_value, shape = [2048,], 是W值
  # Sam: conv_layer_output_value, shape = [7,7,2048]

  pooled_grads_value, conv_layer_output_value = iterate([x])


  # 將 feature map 乘以權重，等於該 feature map 中的某些區域對於該分類的重要性
  for i in range(pooled_grads_value.shape[0]):
    conv_layer_output_value[:, i] *= (pooled_grads_value[i])

  # 計算 feature map 的 channel-wise 加總
  heatmap = np.sum(conv_layer_output_value, axis=-1)

  # 擴充heatmap的維度，方便等等resize成為影像
  heatmap = np.expand_dims(heatmap, axis=0)

  # ReLU
  heatmap = np.maximum(heatmap, 0)

  # 正規化
  heatmap /= np.max(heatmap)

  # 拉伸 heatmap
  heatmap = cv2.resize(heatmap, (x.shape[1], 20))

  heatmap = np.uint8(255 * heatmap)

  # 創建figure和axes
  fig, (ax1,ax2) = plt.subplots(2,1)

  # 繪製曲線圖
  ax1.plot(x[0,:])
  ax1.set_xlim(0, x.shape[1])
  ax1.set_title(y_class_name)

  # 繪製熱力圖
  ax2.imshow(heatmap, cmap='jet')
  ax2.set_title(y_class_name)
  plt.savefig(output_file)
  # files.download(output_file)

  plt.figure
  plt.show()
  plt.close('all')

# 畫出檔案中分類不同的曲線

for i in file_with_diff_cat:
  tmp = data_test['Type'][list(range(file_index[i][0],file_index[i][1]))]
  tmp2 = data_test['FileName'][list(range(file_index[i][0],file_index[i][1]))]

  tmp = tmp.astype('category')
  cat_tmp = tmp.value_counts()
  cat_tmp1 = cat_tmp.index[0]

  for j in range(1,len(cat_tmp)):
    cat_tmp2 = cat_tmp.index[j]

    # 畫出檔案中分類不同的曲線

    plt.plot(data_test.iloc[tmp[tmp==cat_tmp2].index[0],3:data_test.shape[1]])
    plt.title(['voted cat: '+cat_tmp1+'; cat: '+cat_tmp2])
    plt.xlabel('Time')
    plt.ylabel('Temperature Gradient')

    str_1 = str(tmp[tmp==cat_tmp2].index[0])
    output_file = rootpath+'Result/Test_Wrong_Cat/test_' + str_1 + '.jpg'
    plt.savefig(output_file)

    plt.figure
    plt.show()