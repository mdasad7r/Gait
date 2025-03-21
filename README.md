
# 🧍‍♂️ Gait Recognition Using TKAN

This project performs **person identification from gait sequences** using **Temporal Kolmogorov-Arnold Networks (TKAN)**, a neural architecture that models both spatial and temporal patterns from silhouette sequences (CASIA-B dataset).

---

## 📌 About the Network

The model architecture consists of:

- 🧱 **CNNFeatureExtractor**: extracts spatial features from each image
- 🌀 **TKAN**: learns temporal dependencies across walking sequences
- 🎯 **Classifier**: predicts subject identity from learned gait features

> The model takes a **sequence of grayscale images** as input and outputs a subject ID (0–123 for CASIA-B dataset).

---

## 🚀 How to Run This on Google Colab

Follow the steps below to set up and train the model in Colab.

---

### 🔧 Step 1: Clone the Repository

```python
!git clone https://github.com/mdasad7r/Gait
%cd Gait
```

---

### 📦 Step 2: Install Required Libraries

```python
!pip install -r requirement.txt
!pip install tkan
```

---

### 🔐 Step 3: Upload Kaggle Credentials (if using Kaggle for dataset)

```python
from google.colab import files
uploaded = files.upload()  # Upload kaggle.json
```

```python
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

---

### 📁 Step 4: Prepare CASIA-B Dataset

Download and extract CASIA-B dataset into:
```
/content/casia-b/train/output/
                      test/output/
```

Organize it so each subfolder contains gait sequences per subject.

---

### 🏃 Step 5: Train the Model

Make sure to update `config.py` if needed (batch size, epochs, etc.).

```python
!python train.py
```

> Logs will appear in the console and TensorBoard.

---

### 📈 Step 6: Evaluate or Infer

To run evaluation or inference after training:
```python
!python evaluate.py
!python inference.py
```

---

## ✅ Notes on GPU and Memory (Colab Tips)

- If using a **T4/V100**, lower:
  - `BATCH_SIZE = 2`
  - `sequence_len = 32–40`
  - `TKAN units = 128`
- If using an **A100**, you can use:
  - `BATCH_SIZE = 32`
  - `sequence_len = 60`
  - `TKAN units = 512 or 1024`

You can also enable:
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

---

## 🧠 Model Output

The final output is a tensor of shape `(B, 124)`.

Use:
```python
torch.argmax(output, dim=1)
```
To get the predicted subject ID from the gait sequence.

---
