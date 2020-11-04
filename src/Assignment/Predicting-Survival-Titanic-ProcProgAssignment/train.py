import preprocessing_functions as pf
import pandas as pd
import config
# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
df_data = pf.load_data(config.PATH_TO_DATASET)
df_target = df_data[config.TARGET]
df_data = df_data.drop([config.TARGET],axis=1)

# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(df_data,df_target,seed=config.GLOBAL_SEED)

# get first letter from cabin variable
X_train = pf.extract_cabin_letter(X_train,config.IMPUTATION_DICT['cabin_variable'])
X_test = pf.extract_cabin_letter(X_test,config.IMPUTATION_DICT['cabin_variable'])


# impute categorical variables
X_train = pf.add_missing_indicator(X_train,config.CATEGORICAL_VARS)
X_test = pf.add_missing_indicator(X_test,config.CATEGORICAL_VARS)


# impute numerical variable
for var in config.NUMERICAL_TO_IMPUTE:
    X_train = pf.impute_na(X_train,var,replace_by=config.IMPUTATION_DICT[var],add_na_columns=True)
    X_test = pf.impute_na(X_test, var, replace_by=config.IMPUTATION_DICT[var], add_na_columns=True)

# Group rare labels
X_train = pf.remove_rare_labels(X_train,config.FREQUENT_LABELS)
X_test = pf.remove_rare_labels(X_test,config.FREQUENT_LABELS)


# encode categorical variables
for var in config.CATEGORICAL_VARS:
    X_train = pf.encode_categorical(X_train,var)
    X_test = pf.encode_categorical(X_test, var)
X_train.drop(labels=config.CATEGORICAL_VARS,axis=1,inplace=True)
X_test.drop(labels=config.CATEGORICAL_VARS,axis=1,inplace=True)

# check all dummies were added
X_train = pf.check_dummy_variables(X_train,config.DUMMY_VARIABLES)
X_test = pf.check_dummy_variables(X_test,config.DUMMY_VARIABLES)


# train scaler and save
pf.train_scaler(X_train,config.OUTPUT_SCALER_PATH)

# scale train set
X_train = pf.scale_features(X_train,config.OUTPUT_SCALER_PATH)
X_test = pf.scale_features(X_test,config.OUTPUT_SCALER_PATH)


# train model and save
pf.train_model(X_train,y_train,config.OUTPUT_MODEL_PATH,seed=config.GLOBAL_SEED,C=config.NORM_CONSTANT)


print('Finished training')