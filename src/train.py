# 建構訓練模型函式以便單獨與cross-validation使用

def evaluate_model(X_train, y_train, save_model_str, *args):

  # 建構1D-CNN的模型
  model = Sequential()
  # 將模型reshape成(時間長度,1)，每筆輸入的曲線為(時間長度,)的大小
  model.add(Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)))
  model.add(Conv1D(100, 10, activation='relu', input_shape=(X_train.shape[1], 1)))
  model.add(Conv1D(100, 10, activation='relu'))
  model.add(MaxPooling1D(3))
  model.add(Conv1D(160, 10, activation='relu'))
  model.add(Conv1D(160, 10, activation='relu'))
  model.add(GlobalAveragePooling1D())
  #model.add(Dropout(0.5))
  model.add(Dense(y_train.shape[1], activation='softmax'))

  # compile此模型以便開始訓練
  model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])

  if save_model_str == 'model_CV.weights.h5':
    X_val = args[0]
    y_val = args[1]

    # 進入結果資料夾
    os.chdir(rootpath+'Result/Model')
    model.save_weights(save_model_str) #儲存未訓練參數，以便cross-validation重新訓練時重置

    random.seed (21)
    history = model.fit(X_train, y_train, validation_data = (X_val,y_val), epochs=50, batch_size=100, verbose=2)

    _, val_acc = model.evaluate(X_val, y_val, verbose = 1)
    model.load_weights(save_model_str) #讀取未訓練參數，以便cross-validation重新訓練時重置


  else:
    random.seed (21)
    history = model.fit(X_train, y_train, epochs=50, batch_size=100, verbose=2)



    # 進入結果資料夾
    os.chdir(rootpath+'Result/Model')
    model.save_weights(save_model_str) #將訓練好的模型儲存

    _, val_acc = model.evaluate(X_train, y_train, verbose = 1)

  return model, val_acc, history

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# 假設 x_all_test, data_fin, rootpath, 以及已訓練好的 model 均已正確初始化
# 並且 model 是 TF2 的 Sequential 或 Functional 模型

# 將資料轉為符合 Conv1D 要求的形狀： (batch, time_steps, channels)
# 假設原始資料每筆 shape 為 (449,) → 轉換為 (449,1)
# 為了避免 shape 未定義，先用 model.build 指定輸入形狀
model.build((None, 449, 1))
# 或者用一個 dummy 輸入呼叫一次模型，這也會固定輸入形狀
_ = model(tf.zeros((1,449,1)), training=False)

for test_num in range(x_all_test.shape[0]):
    # 格式化編號與輸出檔案路徑
    str_1 = str(test_num).zfill(3)
    output_file = rootpath + 'Result/VisualizationHeatMap_TrainTest/test_' + str_1 + '.jpg'
    print("Test data number:", test_num)

    # 取得測試資料 (假設原始 shape 為 (449,))
    x = x_all_test.iloc[test_num, :].values
    # 轉換成 (1,449)
    x = np.expand_dims(x, axis=0)
    # 再擴展一個維度使得 shape 變為 (1,449,1)
    x = np.expand_dims(x, axis=-1)

    # 取得預測結果與分類
    preds = model.predict(x)
    sorted_indices = np.argsort(-preds, axis=1).squeeze()
    pred_class = sorted_indices[0]

    # 取得分類名稱 (假設 data_fin.Type 為分類欄位)
    y_class_name = data_fin.Type.astype('category').cat.categories[pred_class]

    if test_num == 0:
        print(model.summary())
        print("Selected layer:", model.get_layer("conv1d_31").name)

    # 取得最後一層卷積層 (這裡假設名稱為 "conv1d_31")
    last_conv_layer = model.get_layer("conv1d_31")

    # 再次用 dummy 輸入呼叫模型，確保所有層的 output 已被建立
    _ = model(x, training=False)

    # 建立新的模型，輸出同時包含最後一層卷積層的輸出與模型預測結果
    grad_model = tf.keras.models.Model(inputs=model.inputs,
                                       outputs=[last_conv_layer.output, model.output])

    # 用 tf.GradientTape 計算梯度
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x, training=False)
        loss = predictions[:, pred_class]
    grads = tape.gradient(loss, conv_outputs)

    # 計算每個 channel 的平均梯度作為權重
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # 取出卷積層輸出 (第一個樣本)
    conv_outputs = conv_outputs[0]

    # 將每個 feature map 乘上對應的權重，然後累加
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # 用 ReLU 過濾負值，並正規化 (加上 1e-8 避免除 0)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # 調整熱力圖尺寸，使其寬度等於 x 的特徵數 (449)，高度設為 20
    heatmap = cv2.resize(heatmap, (x.shape[1], 20))
    heatmap = np.uint8(255 * heatmap)

    # 繪製圖形：原始曲線、微分曲線 (示範直接用原始輸入作圖) 與熱力圖
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

    ax1.plot(data_fin.loc[x_all_test.index[test_num]][3:data_fin.shape[1]])
    ax1.set_xlim(0, x.shape[1])
    ax1.set_title(y_class_name)

    ax2.plot(x[0, :, 0])
    ax2.set_xlim(0, x.shape[1])
    ax2.set_title("Differential Curve")

    ax3.imshow(heatmap, cmap='jet', aspect='auto')
    ax3.set_title("Heatmap: " + y_class_name)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close('all')
