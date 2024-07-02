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
![image](https://github.com/saifhmb/Credit-Card-Risk-Assessment-ML-App/assets/111028776/5c7cc327-bc2a-455f-a00d-1ba89ce9c577)


## Evaluation Results
- The target variable, RISK is multiclass. In sklearn, precision and recall functions have a parameter called,
average. This parameter is required for a multiclass/multilabel target. average = 'micro' was used to calculate
the precision and recall metrics globally by counting the total true positives, false negatives and false positives

| Metric    |    Value |
|-----------|----------|
| accuracy  | 0.663957 |
| precision | 0.663957 |
| recall    | 0.663957 |
### Confusion Matrix
![image](https://github.com/saifhmb/Credit-Card-Risk-Assessment-ML-App/assets/111028776/4ec794ed-c965-4fd7-81e7-e0246991a58a)


