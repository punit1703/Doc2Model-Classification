# 🧠 Multiclass Classification from Scratch (NumPy Implementation)

This project demonstrates a **from-scratch implementation of multiclass classification** using **softmax regression** (a generalization of logistic regression), built entirely with **NumPy** — no use of high-level ML libraries like `scikit-learn` for the model.

It is trained and tested on the **Iris dataset**, which includes three classes: `0`, `1`, and `2`.

---

## 🎯 Project Goals

- Implement **softmax regression** for multiclass classification using only NumPy
- Apply the model to real-world data (Iris dataset from `sklearn`)
- Understand how forward pass, softmax activation, cross-entropy loss, and gradients work
- Learn the fundamentals of how libraries like `scikit-learn` and `TensorFlow` build classifiers

---

## 🚀 Features

✅ Implements softmax regression for 3+ classes  
✅ Supports vectorized forward and backward passes  
✅ Feature scaling built-in  
✅ One-hot encoding for targets  
✅ Uses gradient descent to train the model  
✅ Includes evaluation (Accuracy)  
✅ Visualizes predicted outputs using seaborn/matplotlib  
✅ Can be extended to any multiclass dataset  

---

## 📚 Dataset

- **Dataset:** Iris dataset (`sklearn.datasets.load_iris`)
- **Classes:** 3 (Setosa, Versicolor, Virginica)
- **Features:** 4 numerical features per sample

---

## Folder Structure
<!-- TREEVIEW START -->
    ├── implementing_classification/
    │   ├── classification/ # Model package
    │   │   ├── init.py
    │   │   └── model.py # Classification logic 
    │   ├── classification_model_train.py # Script to load data, train, predict, visualize
    │   ├── README.md
    │   ├── .gitignore
    │   └── requirements.txt # NumPy, scikit-learn, matplotlib, seaborn
<!-- TREEVIEW END -->

---
## 🛠️ Installation

```bash
git clone https://github.com/punit1703/Doc2Model-Classification
cd Doc2Model-Classification
pip install -r requirements.txt