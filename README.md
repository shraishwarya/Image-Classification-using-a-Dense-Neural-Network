# 🖼️ Image Classification using a Dense Neural Network

This project implements an **image classification pipeline** using a **fully connected (dense) neural network**. The model is trained to classify images from standard datasets (e.g., **MNIST**, **Fashion-MNIST**, or CIFAR-10 depending on configuration).

The goal is to demonstrate how deep learning can be applied to image classification tasks using a relatively simple architecture, serving as a baseline before experimenting with more advanced models such as CNNs.

---

## 🚀 Features

* Preprocessing of image datasets (flattening + normalization).
* Dense neural network implemented with **TensorFlow / Keras**.
* Configurable **layers, neurons, and activation functions**.
* Training with **cross-entropy loss** and **Adam optimizer**.
* Performance evaluation (accuracy, loss curves).
* Option to easily switch datasets.

---

## 📂 Project Structure

```
Image-Classification-DNN/
│── data/                 # Dataset (downloaded automatically if using Keras datasets)
│── models/               # Saved trained models
│── notebooks/            # Jupyter notebooks for experiments
│── src/
│   ├── data_loader.py    # Dataset loading & preprocessing
│   ├── model.py          # Dense neural network architecture
│   ├── train.py          # Training loop
│   └── evaluate.py       # Model evaluation & metrics
│── requirements.txt      # Project dependencies
│── README.md             # Project documentation
│── main.py               # Entry point to run training & evaluation
```

---

## ⚙️ Installation

1. Clone this repository

   ```bash
   git clone https://github.com/your-username/Image-Classification-DNN.git
   cd Image-Classification-DNN
   ```

2. Create a virtual environment (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### Train the model

```bash
python main.py --mode train --dataset mnist --epochs 20 --batch_size 128
```

### Evaluate the model

```bash
python main.py --mode evaluate --dataset mnist
```

## 📊 Results

Example on **MNIST**:

* Training Accuracy: \~98%
* Test Accuracy: \~97%
  
---

## 🛠️ Tech Stack

* Python 3.8+
* TensorFlow / Keras
* NumPy, Matplotlib, scikit-learn

---

## 📌 Future Improvements

* Implement **Convolutional Neural Networks (CNNs)** for better performance.
* Add **data augmentation** for more robust training.
* Support for custom datasets via directory loaders.
* Deploy trained model as a **web app / API**.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---
