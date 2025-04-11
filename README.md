# CS229: Machine Learning Models in NumPy

This repository contains my personal implementations of machine learning models learned from Stanford's [CS229: Machine Learning](https://cs229.stanford.edu/) course. As I study the lectures, I build each algorithm from scratch in Python (but still using common libraries such as numpy, pandas, etc).

The code is organized by learning paradigm (supervised, unsupervised), with each model in its own directory. Each model's folder includes its own README explaining more about the specific algorithm, its mathematical foundations, and implementation details.

---

## 📁 Repository Structure
```
├── supervised
│   ├── decisiontrees
│   ├── gda
│   ├── L1_norm_linreg
│   ├── local_weight_logreg
│   ├── naivebayes
│   └── svm
└── unsupervised
    ├── kmeans
    └── mix_of_gaussians
```

---

## 🧠 Project Goals

- Build each ML algorithm using python and numpy
- Gain a deep understanding of algorithmic mechanics and math
- Visualize key concepts like decision boundaries, clustering, and EM steps
- Practice clean Python organization and reusable utilities

---

## ✅ Implemented Models

### Supervised Learning
- **Decision Trees**: A classic tree-based model for classification.
- **Random Forests**: An ensemble of decision trees with bootstrap aggregation.
- **AdaBoost**: Boosting method using weak learners.
- **Gaussian Discriminant Analysis (GDA)**: Models classes as multivariate Gaussians.
- **L1-Regularized Linear Regression**: Implements sparsity in parameter space.
- **Locally Weighted Logistic Regression (LWLR)**: Performs logistic regression weighted by proximity.
- **Naive Bayes**: Probabilistic model for classification based on Bayes' theorem.
- **Support Vector Machine (SVM)**: Margin-based classifier with visual decision boundaries.

### Unsupervised Learning
- **K-Means Clustering**: Partitions data into `k` clusters based on proximity.
- **Mixture of Gaussians (EM)**: Fits a probabilistic mixture of Gaussian models via Expectation-Maximization.

### Reinforcement Learning
- *Coming Soon...*

---

## 🛠 Requirements

- Python 3.10+
- `numpy`
- `matplotlib`
- Jupyter Notebook (for exploratory work)