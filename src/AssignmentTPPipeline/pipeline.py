from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline(
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
    [ ('ExtractFirstLetter',pp.ExtractFirstLetter(variables=config.CABIN)),
       ('MissingIndicator',pp.MissingIndicator(variables=config.NUMERICAL_VARS)),
       ('NumericalImputer',pp.NumericalImputer(variables=config.NUMERICAL_VARS)),
       ('CategoricalImputer',pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),
       ('Rarelabel',pp.RareLabelCategoricalEncoder(variables=config.CATEGORICAL_VARS)),
       ('CategoricalEncoder',pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
       ('scaler',StandardScaler()),
      ('model',LogisticRegression(C=config.model_C,random_state=config.RANDOM_STATE))]

    )

