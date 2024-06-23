# importing libraries
from datasets import load_dataset, load_dataset_builder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, classification_report
from transformers import Trainer, TrainingArguments
from skops import hub_utils
import pickle
from skops.card import Card, metadata_from_config
from pathlib import Path
from tempfile import mkdtemp, mkstemp
import streamlit as st
from PIL import Image

# Loading the dataset
dataset_name = "saifhmb/CreditCardRisk"
dataset = load_dataset(dataset_name, split = 'train')
dataset = pd.DataFrame(dataset)
dataset['MARITAL'] = dataset['MARITAL'].str.replace(' ', '')
dataset['MARITAL'] = dataset['MARITAL'].replace(['married', 'single', 'divsepwid'], [0, 1, 2], inplace = True)
dataset['HOWPAID'] = dataset['HOWPAID'].replace(['n', 'y'], [0, 1], inplace = True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding the Independent Variables
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(sparse_output=False), [2, 3, 6, 7])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
#X= X.astype('int')

# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)

# Spliting the datset into Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training Logit Reg Model using the Training set
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting the Test result
y_pred = model.predict(X_test)

# Making the Confusion Matrix and evaluating performance
cm = confusion_matrix(y_pred, y_test, labels=model.classes_)
display_labels = np.array(['bad loss', 'bad profit', 'good risk'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot()
plt.show()
acc = accuracy_score(y_test, y_pred)
ps = precision_score(y_test, y_pred, average ='micro')
rs = recall_score(y_test, y_pred, average ='micro')

# Pickling the model
pickle_out = open("model.pkl", "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()

# Loading the model to predict on the data
pickle_in = open('model.pkl', 'rb') 
model = pickle.load(pickle_in) 

def welcome(): 
    return 'welcome all'

# defining the function which will make the prediction using the data which the user inputs 
def prediction(AGE, INCOME, GENDER, MARITAL, NUMKIDS, NUMCARDS, HOWPAID, MORTGAGE, STORECAR, LOANS):
    prediction = model.predict(sc.transform([[AGE, INCOME, GENDER, MARITAL, NUMKIDS, NUMCARDS, HOWPAID, MORTGAGE, STORECAR, LOANS]]))
    print(prediction)
    return prediction
  
# this is the main function in which we define our webpage  
def main(): 
      # giving the webpage a title 
    st.title("Credit Card Risk Assessment ML App") 
    st.header("Model Description", divider = "gray")
    multi = '''This is a logistic regression model trained on customers' credit card risk dataset in a bank using sklearn library. 
    The model predicts whether a customer is worth issuing a credit card or not.
    For more details on the model please refer to the model card at https://huggingface.co/saifhmb/Credit-Card-Risk-Model
    '''
    st.markdown(multi)
    st.markdown("To determine whether a customer is worth issuing a credit card or not, please **ENTER** the AGE INCOME, GENDER, MARITAL, NUMKIDS, NUMCARDS, HOWPAID, MORTGAGE, STORECAR, and LOANS:")
    col1, col2, col3 = st.columns(3)
    with col1:
        AGE = st.number_input("AGE")
    with col2:
        INCOME = st.number_input("INCOME")
    with col3:
        GENDER = st.text_input("GENDER (Please enter 'm' for male and 'f' for female)")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        MARITAL = st.text_input("MARITAL STATUS (Please enter one of the following options: 'single', 'married', or 'divsepwid')")
    with col5:
        NUMKIDS = st.number_input("Number of dependent children")
    with col6:
        NUMCARDS = st.number_input("Number of credit cards excluding store credit cards")

    col7, col8, col9 =st.columns(3)
    with col7:
        HOWPAID = st.text_input("How often is customer paid by employer (weekly or monthly)")
    with col8:
        MORTGAGE = st.text_input("Does customer have a mortgage? please enter 'y' for yes or 'n' for no")
    with col9:
        STORECAR = st.number_input("Number of store credit cards")

    LOANS = st.number_input("Number of outstanding loans")  
    result = ""
    if st.button("Predict"):
        result = prediction(AGE, INCOME, GENDER, MARITAL, NUMKIDS, NUMCARDS, HOWPAID, MORTGAGE, STORECAR, LOANS)
        if result == 0:
            st.success("The output is {}".format(result) + " which falls under 'bad loss' and thus the customer is NOT worth issuing a credit card")
        if result == 1:
            st.success("The output is {}".format(result) + " which falls under 'bad profit' and thus the customer MAYBE worth issuing a credit card")
        if result == 2:
            st.success("The output is {}".format(result) + " which falls under 'good risk' and thus the customer worth issuing a credit card")

if __name__=='__main__': 
    main() 


