# Machine Learning Algorithms — From Scratch & Scikit-Learn

A hands-on collection of classical ML algorithm implementations with experiments, visualizations, and comparisons. Each notebook explores an algorithm's behaviour by tweaking hyperparameters, checking assumptions, and verifying theoretical properties on real/synthetic datasets.

## Algorithms

### Supervised Learning

| Algorithm | Notebook / Code | Key Experiments |
|---|---|---|
| **Linear Regression** | [`linear_regression/lr.ipynb`](linear_regression/lr.ipynb) | Baseline LR, effect of scaling, polynomial features, adding noise features, multicollinearity check (California Housing) |
| **Ridge & Lasso** | [`linear_regression/ridge_lasso.ipynb`](linear_regression/ridge_lasso.ipynb) | Alpha sweep, coefficient shrinkage paths, Ridge vs Lasso comparison (California Housing) |
| **Logistic Regression** | [`logistic_regression/logistic_regression.ipynb`](logistic_regression/logistic_regression.ipynb) | Baseline LR, regularization with different C values, comparison with KNN (Breast Cancer) |
| **Logistic Regression (from scratch)** | [`logistic_regression/implementation.py`](logistic_regression/implementation.py) | Scratch sigmoid, cross-entropy loss, gradient descent training (Breast Cancer) |
| **K-Nearest Neighbors** | [`knn/knn.ipynb`](knn/knn.ipynb) | Bias-variance tradeoff vs k, scaled vs unscaled, curse of dimensionality, PCA as remedy (Breast Cancer) |
| **Naive Bayes** | [`naive-bayes/nb.ipynb`](naive-bayes/nb.ipynb) | Multinomial NB on SMS spam data, comparison with Logistic Regression, threshold tuning, top features |
| **Support Vector Machines** | [`svm/svm.ipynb`](svm/svm.ipynb) | Linear & RBF kernels, C (capacity control), gamma tuning, decision boundary plots (make_moons) |
| **Decision Trees** | [`trees/decision_trees.ipynb`](trees/decision_trees.ipynb) | Depth vs accuracy, bias-variance, scaling invariance, regression trees, decision boundary visualization, cost-complexity pruning, feature importances (Breast Cancer) |
| **Random Forest** | [`trees/rf.ipynb`](trees/rf.ipynb) | DT vs RF comparison, hyperparameter tuning, OOB score analysis (Breast Cancer) |
| **Gradient Boosting** | [`trees/boosting.ipynb`](trees/boosting.ipynb) | DT vs RF vs GBR, learning rate experiments, small LR + more trees vs big LR + fewer trees (synthetic data) |

### Unsupervised Learning

| Algorithm | Notebook | Key Experiments |
|---|---|---|
| **PCA** | [`pca/pca.ipynb`](pca/pca.ipynb) | Geometric & variance intuition, explained variance, KNN+PCA, Decision Tree+PCA (synthetic data) |
| **K-Means Clustering** | [`clustering/kmeans.ipynb`](clustering/kmeans.ipynb) | K vs inertia (elbow method), effect of scaling, failure on non-spherical clusters (make_moons), initialization methods |

## Repository Structure

```
├── linear_regression/
│   ├── lr.ipynb              # Linear Regression experiments
│   └── ridge_lasso.ipynb     # Ridge & Lasso regularization
├── logistic_regression/
│   ├── logistic_regression.ipynb  # Sklearn Logistic Regression
│   └── implementation.py         # From-scratch implementation
├── knn/
│   └── knn.ipynb             # K-Nearest Neighbors
├── naive-bayes/
│   ├── nb.ipynb              # Naive Bayes (spam classification)
│   └── spam.csv              # SMS spam dataset
├── svm/
│   └── svm.ipynb             # Support Vector Machines
├── trees/
│   ├── decision_trees.ipynb  # Decision Trees (classification & regression)
│   ├── rf.ipynb              # Random Forest
│   └── boosting.ipynb        # Gradient Boosting
├── pca/
│   └── pca.ipynb             # Principal Component Analysis
├── clustering/
│   └── kmeans.ipynb          # K-Means Clustering
├── requirements.txt
└── README.md
```

## Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/algorithms_implementation.git
cd algorithms_implementation

# Install dependencies
pip install -r requirements.txt

# Open any notebook
jupyter notebook
```

## Requirements

- Python 3.8+
- numpy, pandas, matplotlib, seaborn, scikit-learn (see [`requirements.txt`](requirements.txt))

## Notes

- Notebooks contain inline commentary explaining *why* each experiment is run and what the expected outcome is.
- Most notebooks use scikit-learn datasets (`load_breast_cancer`, `fetch_california_housing`, `make_moons`, etc.) so no extra data downloads are needed — the only external file is `naive-bayes/spam.csv`.
- Results may vary slightly based on random seeds.
