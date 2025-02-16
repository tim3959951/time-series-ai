# 📊 AI-Powered Time-Series Classification  

This project applies a **1D Convolutional Neural Network (1D-CNN)** to classify **time-series data** from an industrial dataset.  
The model is trained to distinguish between different operational conditions, with a focus on **anomaly detection** and **predictive maintenance**.  

---

## 🚀 Project Overview  

- **Dataset**: Industrial sensor readings from **IMBD AI & Big Data Competition**  
- **Model**: 1D-CNN with **Conv1D**, **MaxPooling**, and **GlobalAveragePooling**  
- **Techniques**: K-Fold Cross-Validation, Data Augmentation, Grad-CAM for interpretability  
- **Performance**: **90.9% average accuracy**, with best model achieving **91.2% accuracy**  
- **Deployment**: Future goal to optimise for **real-time monitoring & edge inference**  

---

## 📂 Project Structure  

📂 time-series-ai │── 📂 src # Core scripts
│ ├── data_loader.py # Load & preprocess time-series data
│ ├── train.py # Train 1D-CNN model
│ ├── evaluate.py # Evaluate model (accuracy, confusion matrix)
│ ├── visualization.py # Generate Grad-CAM heatmaps & result plots
│── 📂 experiments # Training notebooks & performance analysis
│── time_series_classification.ipynb # Full Jupyter Notebook for model training
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│── .gitignore # Ignore unnecessary files


---

## 🔬 Data Processing  

- **Source**: Time-series sensor data from **thubigdata2019training-230**  
- **Preprocessing**:  
  - Removed unit labels (Deg.F)  
  - Converted all values to `float`  
  - Normalised time-series data  
  - **Total records**: **1,745**, with **1,723 valid samples**  

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
  - **Best accuracy**: **91.2%**  
  - **Average accuracy**: **90.9%**  

---

## 📈 Model Performance  

✅ **Confusion Matrix**: Visualises model accuracy across all classes  
✅ **Grad-CAM Heatmaps**: Highlights key areas influencing model decisions  
✅ **Prediction vs. True Labels**: Evaluates model reliability  

Example Confusion Matrix:

```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
```

---
## 🔥 Results & Key Findings  

📌 **Best Model**:  
- **1D-CNN (Conv1D + MaxPooling + GlobalAvgPooling)**  
- **Achieved 89.2% accuracy on validation set**  

📌 **What Worked Well**:  
- **Cross-validation** improved robustness  
- **Grad-CAM** helped explain AI decisions  
- **Data augmentation** enhanced generalisation  

📌 **Future Improvements**:  
- Implement **self-supervised learning** for better feature extraction  
- Deploy model on **embedded devices** for real-time monitoring  

---

## 📌 How to Run  

1️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```
1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
2️⃣ **Train the model**
```bash
python src/train.py
```
3️⃣ **Evaluate the model**
```bash
python src/evaluate.py
```
4️⃣ **Visualise results**
```bash
python src/visualization.py
```

