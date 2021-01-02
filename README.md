# Machine learning exam
Exam project comparing two different machine learning models (KNN and AE) on a dataset of fraudlent and legitimate credit card transactions.

# Data
You will need to download the data from: https://www.kaggle.com/mlg-ulb/creditcardfraud, and place it in the data folder to run the project.

# Results
KNN
```
[[5940   60]
 [  70  372]]
              precision    recall  f1-score   support

         0.0       0.99      0.99      0.99      6000
         1.0       0.86      0.84      0.85       442

    accuracy                           0.98      6442
   macro avg       0.92      0.92      0.92      6442
weighted avg       0.98      0.98      0.98      6442

AU-PRC: 0.73560223888949
baseline: 0.06861223222601677
[[5489  511]
 [  62  380]]
              precision    recall  f1-score   support

         0.0       0.99      0.91      0.95      6000
         1.0       0.43      0.86      0.57       442

    accuracy                           0.91      6442
   macro avg       0.71      0.89      0.76      6442
weighted avg       0.95      0.91      0.92      6442

AU-PRC: 0.37628745202814884
baseline: 0.06861223222601677
```