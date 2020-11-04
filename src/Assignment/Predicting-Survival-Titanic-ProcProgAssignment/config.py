# ====   PATHS ===================

PATH_TO_DATASET = "titanic.csv"
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'logistic_regression.pkl'
GLOBAL_SEED = 0
NORM_CONSTANT = 0.0005
# ======= PARAMETERS ===============

# imputation parameters
IMPUTATION_DICT = {'cabin_variable':'cabin',
                   'age':28.0,
                   'fare':14.4542}


# encoding parameters
FREQUENT_LABELS = {'sex':['male','female'],
                   'cabin': ['Missing','C'],
                   'embarked':['S','C','Q'],
                   'title':['Mr','Miss','Mrs']}


DUMMY_VARIABLES = ['cabin_Missing','cabin_Rare','embarked_Rare','title_Rare']


# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex','cabin','embarked','title']

NUMERICAL_TO_IMPUTE = ['age', 'fare']