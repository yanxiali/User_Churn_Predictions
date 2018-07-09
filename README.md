# User_Churn_Prediction

**Results**:
Please check out the jupyter notebook of the project [here](https://github.com/yanxiali/User_Churn_Predictions/blob/master/User_Churn_Prediction.ipynb). If you experience loading problems (as it is a big file), please take a look of a markdown copy of the project [here](https://github.com/yanxiali/User_Churn_Predictions/blob/master/results/User_Churn_Prediction.md)

**Keywords**:
11 supervised learning models - ensemble modeling - learning curves and confusion matrices - thereshold determination - feature importance - recursive feature elimination

**Description**:
In this project, I used supervised learning models to identify cell phone service customers who are more likely to stop using the service in the future and created a model that can predict if a certain customer will drop the service. Furthermore, I analyzed top factors that influence user retention to help the company prevent user churn.The dataset contains the information of customers' plans and usage of the service, as well as whether or not they stopped using the service eventually. 
- Data Source: https://www.sgi.com/tech/mlc/db/churn.all  
- Data info: https://www.sgi.com/tech/mlc/db/churn.names

I compared 11 of the most popular classifiers and evaluated their performance using a stratified K-fold cross validation procedure. I selected 6 classifiers with good cross validation scores covering a wide variety of types (e.g., tree-based, regression). I then performed hyper-parameter tuning on these models. I examined the learning curves and confusion matrices of different classifiers and I found that the XGBoost has a AUROC score of 0.91 and better generalizes the prediction without over-fitting the training data. 

Furthermore, I derived the feature importance from 3 tree based classifiers and I noticed that these three classifiers have different top features according to the relative importance. It means that their predictions are not based on the same features. Nevertheless, they share some common important features for the classification, e.g., day/night call charges. 

Instead of using the default discrimination threshold of 0.5, I determined the optimal threshold by examining the ROC curves and how different evaluation metrics change with various thresholds. Given that our goal is to minimize the loss of profitability, I will provide a special offer to the customers who will likely drop the service. I found that the loss is minimized at a threshold of 0.35. With the new threshold I got a testing score of 0.91 (AUROC).
