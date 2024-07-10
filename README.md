---
title: Credit Card Risk Assessment ML App
emoji: 🏆
colorFrom: green
colorTo: indigo
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
---
# Credit-Card-Risk-Assessment-ML-App
Hugging Face ML Deployment of Streamlit App for a Credit Card Risk Assessment Model https://huggingface.co/spaces/saifhmb/Credit-Card-Risk-Assessment-ML-App

# Model description

This is a logistic regression model trained on customers' credit card risk data in a bank using sklearn library.
The model predicts whether a customer is worth issuing a credit card or not. The full dataset can be viewed at the following link: https://huggingface.co/datasets/saifhmb/CreditCardRisk


## Training Procedure

The data preprocessing steps applied include the following:
- Dropping high cardinality features, specifically ID
- Transforming and Encoding categorical features namely: GENDER, MARITAL, HOWPAID, MORTGAGE and the target variable, RISK
- Splitting the dataset into training/test set using 85/15 split ratio
- Applying feature scaling on all features
### Model Plot
![image](https://github.com/saifhmb/Credit-Card-Risk-Assessment-ML-App/assets/111028776/99b017ed-4771-462e-8e15-37e454b983be)



## Evaluation Results
- The target variable, RISK is multiclass. In sklearn, precision and recall functions have a parameter called,
average. This parameter is required for a multiclass/multilabel target. average = 'micro' was used to calculate
the precision and recall metrics globally by counting the total true positives, false negatives and false positives

| Metric    |    Value |
|-----------|----------|
| accuracy  | 0.699187 |
| precision | 0.699187 |
| recall    | 0.699187 |

### Feature Importance
SHAP was used to determine the important features that helps the model make decisions
![image](https://github.com/saifhmb/Credit-Card-Risk-Assessment-ML-App/assets/111028776/b25f372b-f024-4d4c-96b9-187bcdb37a57)


### Confusion Matrix
![image](https://github.com/saifhmb/Credit-Card-Risk-Assessment-ML-App/assets/111028776/66449d90-6a41-4510-93b7-56e496e13140)


