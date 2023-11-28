<h1 align="center">🤖ML Fundamentals</h1>
<p align="center">The following details and explains performing classification on the Iris dataset using a range of ML models:</br>
k-NNs, Naive Bayes, Stochastic Gradient Descent, Decision Trees, Random Forest, SVM, Logistic Regression, Neural Nets</br></br>
I similarly do the same for regression on the USA House Pricing dataset using:</br>Linear Regression (and soon Polynomial Regression, Support Vector Regression (SVR), Random Forest Regression, Regularised regression models (Ridge, Lasso)</p>

# 🌼Classification on Iris Dataset
All content for classification on the Iris dataset can be found in the following notebook:

[Notebook](https://github.com/dilne/ML-Fundamentals/blob/main/Notebooks/Iris%20Classification.ipynb)

<a href="https://colab.research.google.com/github/dilne/ML-Fundamentals/blob/main/Notebooks/Iris%20Classification.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Dataset
The iris dataset comprises three different irises each with 3 different features, petal length, petal width, sepal width, and sepal length. Some EDA is performed using a simple pairplot to provide an indication of any groupings in the dataset.

<div align="center">
  <a href="https://github.com/dilne/ML-Fundamentals/blob/main/Images/Iris/Pairplot.png" target="_blank">
    <img src="https://github.com/dilne/ML-Fundamentals/blob/main/Images/Iris/Pairplot.png" alt="Iris Pairplot" style="height:200px;"/>
  </a>
</div>

## 1⃣️k-NN
Using k-Nearest Neighbours to predict the iris type using features of the plant

### Optimimum k

When using a k-NN model, it can be useful to find the optimum k number. The model is trained for a k from 1 to 40 and it was found that the best k number for this dataset is 4.
<div align="center">
  <a href="https://github.com/dilne/ML-Fundamentals/blob/main/Images/Iris/K.png" target="_blank">
    <img src="https://github.com/dilne/ML-Fundamentals/blob/main/Images/Iris/K.png" alt="Finding the best K" style="height:200px;"/>
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

## 2⃣️Naive Bayes
Using Naive Bayes to predict the iris type using features of the plant

### Results
The accuracy of the Naive Bayes model is high, but is not perfect:
```
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        19
           2       1.00      0.92      0.96        13
           3       0.93      1.00      0.96        13

    accuracy                           0.98        45
   macro avg       0.98      0.97      0.97        45
weighted avg       0.98      0.98      0.98        45
```

## 3⃣️Stochastic Gradient Descent
Using Stochastic Gradient Descent to predict the iris type using features of the plant

### Results
```
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        19
           2       1.00      0.69      0.82        13
           3       0.76      1.00      0.87        13

    accuracy                           0.91        45
   macro avg       0.92      0.90      0.89        45
weighted avg       0.93      0.91      0.91        45
```

## 4⃣️Decision Tree
Using a Decision Tree to predict the iris type using features of the plant

### Results
```
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        19
           2       1.00      1.00      1.00        13
           3       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
```

## 5⃣️Random Forest
Using a Random Foreste to predict the iris type using features of the plant

### Results
```
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        19
           2       1.00      1.00      1.00        13
           3       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
```

## 6⃣️Support Vector Machine (SVM)
Using an SVM to predict the iris type using features of the plant

### Results
```
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        19
           2       1.00      1.00      1.00        13
           3       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
```

## 7⃣️Logistic Regression
Using Logistic Regression to predict the iris type using features of the plant

### Results
```
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        19
           2       1.00      1.00      1.00        13
           3       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
```

## 8⃣️lbfgs Neural Network
Using a Neural Network with an lbfgs optimsed to predict the iris type using features of the plant

### Results
```
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        19
           2       0.93      1.00      0.96        13
           3       1.00      0.92      0.96        13

    accuracy                           0.98        45
   macro avg       0.98      0.97      0.97        45
weighted avg       0.98      0.98      0.98        45
```

# 🏡Linear Regression on USA Housing Dataset
Using linear regression to predict house prices on the USA Housing Dataset

[Notebook](https://github.com/dilne/ML-Fundamentals/blob/main/Notebooks/Linear%20Regression%20-%20USA%20Housing.ipynb)

<a href="https://colab.research.google.com/github/dilne/ML-Fundamentals/blob/main/Notebooks/Linear%20Regression%20-%20USA%20Housing.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Dataset
This USA housing dataset is a limited version of the official dataset. It contains 5,000 samples of house prices with their corresponding Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, Area Population, and Address.

<div align="center">
  <a href="https://github.com/dilne/ML-Fundamentals/blob/main/Images/USA%20Housing/Pairplot.png" target="_blank">
    <img src="https://github.com/dilne/ML-Fundamentals/blob/main/Images/USA%20Housing/Pairplot.png" alt="USA Housing Pairplot" style="height:200px;"/>
  </a>
</div>

### Results
The linear regression model worked fairly well, providing a high R^2 score and low MAE percentage:
```
R^2 Score: 0.9179971706834331
Mean Absolute Error: 80879.09723489445
Mean Squared Error: 10089009300.893993
Root Mean Squared Error: 100444.06055558483
Mean Absolute Percentage Error: 7.3878388597543685%
```

<div align="center">
  <a href="https://github.com/dilne/ML-Fundamentals/blob/main/Images/USA%20Housing/Predictions.png" target="_blank">
    <img src="https://github.com/dilne/ML-Fundamentals/blob/main/Images/USA%20Housing/Predictions.png" alt="Linear Regression" style="height:200px;"/>
  </a>
</div>

