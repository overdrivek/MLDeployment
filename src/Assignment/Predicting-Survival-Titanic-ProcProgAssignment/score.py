import preprocessing_functions as pf
import config

# =========== scoring pipeline =========

# impute categorical variables
def predict(data):
    
    # extract first letter from cabin
    X_test = pf.extract_cabin_letter(data, config.IMPUTATION_DICT['cabin_variable'])

    # impute NA categorical
    X_test = pf.add_missing_indicator(X_test, config.CATEGORICAL_VARS)
    
    
    # impute NA numerical
    for var in config.NUMERICAL_TO_IMPUTE:
        X_test = pf.impute_na(X_test,var,replace_by='median', add_na_columns=True)

    
    # Group rare labels
    X_test = pf.remove_rare_labels(X_test, config.FREQUENT_LABELS)
    
    # encode variables
    for var in config.CATEGORICAL_VARS:
        X_test = pf.encode_categorical(X_test, var)
    X_test.drop(labels=config.CATEGORICAL_VARS, axis=1, inplace=True)
        
    # check all dummies were added
    X_test = pf.check_dummy_variables(X_test, config.DUMMY_VARIABLES)

    
    # scale variables
    X_test = pf.scale_features(X_test, config.OUTPUT_SCALER_PATH)
    
    # make predictions
    predictions = pf.predict(X_test,config.OUTPUT_MODEL_PATH)

    
    return predictions

# ======================================
    
# small test that scripts are working ok
    
if __name__ == '__main__':
        
    from sklearn.metrics import accuracy_score    
    import warnings
    warnings.simplefilter(action='ignore')
    
    # Load data
    data = pf.load_data(config.PATH_TO_DATASET)
    df_target = data[config.TARGET]
    data = data.drop([config.TARGET],axis=1)

    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            df_target,seed=config.GLOBAL_SEED)
    
    pred = predict(X_test)
    
    # evaluate
    # if your code reprodues the notebook, your output should be:
    # test accuracy: 0.6832
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()
        