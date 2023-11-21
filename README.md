<h1 align="center">🤖ML Fundamentals</h1>
<p align="center">k-NNs, Linear Regression, Logistic Regression, SVM, Decision Trees, Random Forest Regressor, XGBOOST, Naive Bayes</p>

## 🌼k-NN on Iris Dataset
Using k-Nearest Neighbours to predict the iris type using features of the plant

[Notebook](https://github.com/dilne/ML-Fundamentals/blob/main/k-NN%20-%20Iris/k-NN%20-%20Iris.ipynb)

<a href="https://colab.research.google.com/github/dilne/ML-Fundamentals/blob/main/k-NN%20-%20Iris/k-NN%20-%20Iris.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Dataset
The iris dataset comprises three different irises each with 3 different features, petal length, petal width, sepal width, and sepal length. Some EDA is performed using a simple pairplot to provide an indication of any groupings in the dataset.

<div align="center">
  <a href="https://github.com/dilne/ML-Fundamentals/blob/main/k-NN%20-%20Iris/Pairplot.png" target="_blank">
    <img src="https://github.com/dilne/ML-Fundamentals/blob/main/k-NN%20-%20Iris/Pairplot.png" alt="k-NN Pairplot" style="height:200px;"/>
  </a>
</div>

### Optimimum k

When using a k-NN model, it can be useful to find the optimum k number. The model is trained for a k from 1 to 40 and it was found that the best k number for this dataset is 4.
<div align="center">
  <a href="https://github.com/dilne/ML-Fundamentals/blob/main/k-NN%20-%20Iris/K.png" target="_blank">
    <img src="https://github.com/dilne/ML-Fundamentals/blob/main/k-NN%20-%20Iris/K.png" alt="Finding the beset K" style="height:200px;"/>
  </a>
</div>

### Results
The model with a k of 4 achieved perfect accuracy:

```
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       1.00      1.00      1.00        19
 Iris-virginica       1.00      1.00      1.00        15

       accuracy                           1.00        45
      macro avg       1.00      1.00      1.00        45
   weighted avg       1.00      1.00      1.00        45
```
<div align="center">
  <a href="https://github.com/dilne/ML-Fundamentals/blob/main/k-NN%20-%20Iris/Confusion%20matrix.png" target="_blank">
    <img src="https://github.com/dilne/ML-Fundamentals/blob/main/k-NN%20-%20Iris/Confusion%20matrix.png" alt="Finding the beset K" style="height:200px;"/>
  </a>
</div>

## 🌼Naive Bayes on Iris Dataset
Using Naive Bayes to predict the iris type using features of the plant

[Notebook](https://github.com/dilne/ML-Fundamentals/blob/main/Naive%20Bayes%20-%20Iris.ipynb)

<a href="https://colab.research.google.com/github/dilne/ML-Fundamentals/blob/main/Naive%20Bayes%20-%20Iris.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Dataset
The iris dataset comprises three different irises each with 3 different features, petal length, petal width, sepal width, and sepal length. Some EDA is performed using a simple pairplot to provide an indication of any groupings in the dataset.

<div align="center">
  <a href="https://github.com/dilne/ML-Fundamentals/blob/main/Naive%20Bayes%20-%20Iris/Pairplot.png" target="_blank">
    <img src="https://github.com/dilne/ML-Fundamentals/blob/main/Naive%20Bayes%20-%20Iris/Pairplot.png" alt="k-NN Pairplot" style="height:200px;"/>
  </a>
</div>

### Results
The accuracy of the Naive Bayes model is high, but is not perfect like the k-NN model used in the above notebook:
```
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        19
           2       1.00      0.92      0.96        13
           3       0.93      1.00      0.96        13

    accuracy                           0.98        45
   macro avg       0.98      0.97      0.97        45
weighted avg       0.98      0.98      0.98        45
```
<div align="center">
  <a href="https://github.com/dilne/ML-Fundamentals/blob/main/Naive%20Bayes%20-%20Iris/Confusion%20matrix.png" target="_blank">
    <img src="https://github.com/dilne/ML-Fundamentals/blob/main/Naive%20Bayes%20-%20Iris/Confusion%20matrix.png" alt="Finding the beset K" style="height:200px;"/>
  </a>
</div>

## 🏡Linear Regression on USA Housing Dataset
Using linear regression to predict house prices on the USA Housing Dataset

[Notebook](https://github.com/dilne/ML-Fundamentals/blob/main/Linear%20Regression%20-%20USA%20Housing.ipynb)

<a href="https://colab.research.google.com/github/dilne/ML-Fundamentals/blob/main/Linear%20Regression%20-%20USA%20Housing.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
