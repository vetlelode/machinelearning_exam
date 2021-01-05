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
Typical result of an execution of the KNN code:
![KNN image](img/knn.png)
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
## Autoencoder
![KNN image](img/Ae-LL.png)

```
r2 report:
[[116031   3576]
 [   132    317]]
              precision    recall  f1-score   support

         0.0       1.00      0.97      0.98    119607
         1.0       0.08      0.71      0.15       449

    accuracy                           0.97    120056
   macro avg       0.54      0.84      0.57    120056
weighted avg       1.00      0.97      0.98    120056

AU-PRC:   0.12034870433061656
baseline: 0.0037399213700273206
Threshold: 1.0290043293222664
Optimal threshold: 1.090871774622069

AE-LL report:
[[117753   1854]
 [    88    361]]
              precision    recall  f1-score   support

         0.0       1.00      0.98      0.99    119607
         1.0       0.16      0.80      0.27       449

    accuracy                           0.98    120056
   macro avg       0.58      0.89      0.63    120056
weighted avg       1.00      0.98      0.99    120056

AU-PRC:   0.32352661300064983
baseline: 0.0037399213700273206
Threshold: 232.80144134050897
Optimal threshold: 265.8333236663422

direct-LL report:
[[118254   1353]
 [   166    283]]
              precision    recall  f1-score   support

         0.0       1.00      0.99      0.99    119607
         1.0       0.17      0.63      0.27       449

    accuracy                           0.99    120056
   macro avg       0.59      0.81      0.63    120056
weighted avg       1.00      0.99      0.99    120056

AU-PRC:   0.2614251422041023
baseline: 0.0037399213700273206
Threshold: 186.50434548160416
Optimal threshold: 138.69541913655996
```
## Average AURPC scores on 20 iterations:
| Support | Baseline | AE R²       | AE Log-likelihood | PureLog-likelihood | Mixed sample KNN; K = 10 | Inlier Only KNN; K=20 |
|---------|----------|-------------|-------------------|--------------------|--------------------------|-----------------------|
| 120492  | 0.0041   | 0.099BF 27  | 0.375 BF 145      | 0.310BF 109        | 0.706BF 588              | 0.457                 |
| 70492   | 0.007    | 0.172 BF 29 | 0.460BF 120       | 0.407BF 97         | 0.738                    | 0.515                 |
| 18467   | 0.027    | 0.387BF 22  | 0.706BF 86        | 0.664 BF 71        | 0.773                    | 0.717                 |
| 6467    | 0.072    | 0.611BF 20  | 0.831 BF 63       | 0.811BF 55         | 0.778                    | 0.822BF 65            |
