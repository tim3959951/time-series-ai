# 📊 AI-Powered Time-Series Classification  

This project applies a **1D Convolutional Neural Network (1D-CNN)** to classify **time-series data** from an industrial dataset.  
The model is trained to distinguish between different operational conditions, with a focus on **anomaly detection** and **predictive maintenance**.  

---

## 🚀 Project Overview  

- **Dataset**: Industrial sensor readings from **IMBD AI & Big Data Competition**  
- **Model**: 1D-CNN with **Conv1D**, **MaxPooling**, and **GlobalAveragePooling**  
- **Techniques**: K-Fold Cross-Validation, Data Augmentation, Grad-CAM for interpretability  
- **Performance**: **99.8% average accuracy**, with best model achieving **100.0% accuracy**  
- **Deployment**: Future goal to optimise for **real-time monitoring & edge inference**  

---

## 📂 Project Structure  

| File/Folder             | Description                                      |
|-------------------------|--------------------------------------------------|
| 📂 src                 | Contains all core scripts                        |
| 📂 visualizations      | Stores generated plots & Grad-CAM heatmaps       |
| 📄 requirements.txt    | Python dependencies                              |
| 📄 README.md           | Project documentation                           |
| 📄 .gitignore          | Ignore unnecessary files                         |
| 📄 2019_IMBD_ShuJiBao_Vis_1D_CNN_final.ipynb | Main Jupyter Notebook |




---


## 🔬 Data Processing  

- **Dataset**: Sensor time-series data from **thubigdata2019training-230**  
- **Key Preprocessing Steps**:  
  ✅ **Standardised time-series values** for consistency  
  ✅ **Converted categorical labels to one-hot encoding**  
  ✅ **Applied Min-Max Scaling** to normalise feature distributions  
  ✅ **Removed invalid or missing records** (final dataset: **1,723 samples**)  


---

## 🏗 Model Architecture  

The model is a **1D-CNN** designed for time-series classification:  

```python
model = Sequential()
model.add(Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)))  
model.add(Conv1D(100, 10, activation='relu'))  
model.add(Conv1D(100, 10, activation='relu'))  
model.add(MaxPooling1D(3))  
model.add(Conv1D(160, 10, activation='relu'))  
model.add(Conv1D(160, 10, activation='relu'))  
model.add(GlobalAveragePooling1D())  
model.add(Dense(y_train.shape[1], activation='softmax'))
Conv1D Layers: Extract temporal patterns
MaxPooling: Downsampling for efficiency
GlobalAveragePooling: Feature compression
Softmax Output: Multi-class classification
```

---

## 🎯 Training Strategy  

- **Loss Function**: `categorical_crossentropy` (multi-class classification)  
- **Optimizer**: Adam  
- **K-Fold Cross-Validation**: 5-fold  
- **Training Setup**:  
  - `epochs=50`, `batch_size=100`  
  - **Best accuracy**: **100.0%**  
  - **Average accuracy**: **99.8%**  

---
## 📈 Model Performance  

✅ **Confusion Matrix**: Evaluates classification accuracy across all classes  
✅ **Grad-CAM Heatmaps**: Highlights key areas influencing model decisions  
✅ **Prediction vs. True Labels**: Assesses model reliability  

### Confusion Matrices  
| Training Set | Test Set |
|-------------|---------|
| ![Train Confusion Matrix](visualizations/train_confusion_matrix.jpg) | ![Test Confusion Matrix](visualizations/test_confusion_matrix.jpg) |

Example Code:  
```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
```

---
## 🔥 Results & Key Findings  

📌 **Model Performance**:  
- **1D-CNN (Conv1D + MaxPooling + GlobalAvgPooling)**  
- **Achieved 99.8% average accuracy (Cross-Validation), with best accuracy reaching 100.0%**  
- **Official Test Accuracy (Competition Evaluation): 99.0%**  

📌 **What Worked Well**:  
- **Cross-validation** improved model robustness and reduced overfitting.  
- **Grad-CAM** provided interpretability by highlighting important regions in the time-series data.  

📌 **Limitations & Future Directions**:  
- 🔹 **This project focused on 1D-CNN without comparing alternative architectures like LSTM or Transformers.**  
- 🔹 **Future work may explore recurrent models (e.g., LSTM, GRU) or Transformer-based approaches to enhance time-series feature extraction.**  
- 🔹 **Investigate real-world deployment feasibility on embedded devices.**  



