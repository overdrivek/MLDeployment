import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    df_return = pd.read_csv(df_path,sep=',')
    return df_return



def divide_train_test(df, target,seed=0):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test =train_test_split(df,target,test_size=0.2,random_state=seed)
    return X_train,X_test,y_train,y_test

    



def extract_cabin_letter(df, var):
    # captures the first letter
    df[var]=df[var].apply(lambda x: x[0] if not pd.isnull(x) else x)
    return df





def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    for v in var:
        df[v] = df[v].fillna('Missing')
    return df


    
def impute_na(df_train,var,replace_by='Missing', add_na_columns=True):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)

    if add_na_columns is True:
        df_train[var+'_na'] = np.where(df_train[var].isnull(),1,0)

        if replace_by == 'mean':
            replace_value = df_train[var].mean()
        elif replace_by == 'median':
            replace_value = df_train[var].median()
        elif replace_by == 'mode':
            replace_value = df_train[var].mode()
        else:
            replace_value = 'Missing'
    df_train[var] = df_train[var].fillna(replace_value)

    return df_train



def remove_rare_labels(df,frequent_labels):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    for var,frequent_vals in frequent_labels.items():
        df[var] = np.where(df[var].isin(frequent_vals),df[var],'Rare')
    return df



def encode_categorical(df, var):
    # adds ohe variables and removes original categorical variable
    
    df = df.copy()
    df = pd.concat([df,pd.get_dummies(df[var],prefix=var,drop_first=True)],axis=1)
    return df



def check_dummy_variables(df, dummy_list):
    
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    for dummy_var in dummy_list:
        if dummy_var not in df.columns:
            df[dummy_var] = 0

    return df
    

def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)

  
    

def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path)  # with joblib probably
    return scaler.transform(df)



def train_model(df, target, output_path,seed=0, C=0):
    # train and save model
    classifier = LogisticRegression(random_state=seed, C=C)
    classifier.fit(df,target)

    joblib.dump(classifier, output_path)

    return classifier

def predict(df, model):
    # load model and get predictions
    classifier = joblib.load(model)
    return classifier.predict(df)

