# Machine learning exam
Exam project comparing two different machine learning models (KNN and AE) on a dataset of fraudlent and legitimate credit card transactions.

# Data
You will need to download the data from: https://www.kaggle.com/mlg-ulb/creditcardfraud, and place it in the data folder to run the project. On the first ever execution, the preprocessing code will do all the nesecarry changes to this file and place the results under the data folder

# Running the project:
## KNN
    The KNN model has to be run from the src folder, from there run the knndist.py file either in and IDE or from the terminal.

## Autoencoder
    Like KNN, but run the file autoencoder.py

***Note that both models have to be run from inside the SRC folder if you are executing from the terminal***
# Results
## KNN
```
predicting outliers based on knn classes
predicting outliers based on knn outliers scores
[[69901    11]
 [  131   318]]
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     69912
         1.0       0.97      0.71      0.82       449

    accuracy                           1.00     70361
   macro avg       0.98      0.85      0.91     70361
weighted avg       1.00      1.00      1.00     70361

AU-PRC: 0.6864225866750382
baseline: 0.006381376046389335
[[69467   445]
 [   96   353]]
              precision    recall  f1-score   support

         0.0       1.00      0.99      1.00     69912
         1.0       0.44      0.79      0.57       449

    accuracy                           0.99     70361
   macro avg       0.72      0.89      0.78     70361
weighted avg       1.00      0.99      0.99     70361

AU-PRC: 0.5871356710114534
baseline: 0.006381376046389335
Threshold: 5.453659516922431
Optimal threshold: 7.021158787236504
```