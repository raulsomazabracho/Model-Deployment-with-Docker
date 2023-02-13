# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
import sklearn

insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
insurance

insurance.describe()

"""## Normalization"""

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import sklearn
import warnings

# Commented out IPython magic to ensure Python compatibility.
def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                  % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names

# Transform our columns
ct = make_column_transformer(
  (MinMaxScaler(), ['age']),
  (MinMaxScaler(), ['bmi']),
  (MinMaxScaler(), ['children']),
  (OneHotEncoder(handle_unknown='ignore'),['sex', 'smoker', 'region'])
)


# X&Y
X = insurance.drop('charges', axis=1)
y = insurance['charges']

#  Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

#Fit The column transformer
ct.fit(X_train)

get_feature_names(ct)

# Gert Normalization variables
age_scaler = ct.named_transformers_['minmaxscaler-1']
age_scaler_min = age_scaler.data_min_
age_scaler_max = age_scaler.data_max_

bmi_scaler = ct.named_transformers_['minmaxscaler-2']
bmi_scaler_min = bmi_scaler.data_min_
bmi_scaler_max = bmi_scaler.data_max_

ch_scaler = ct.named_transformers_['minmaxscaler-3']
ch_scaler_min = ch_scaler.data_min_
ch_scaler_max = ch_scaler.data_max_

# Save the normalization constants to a file
with open('model_normalization_constants.pkl', 'wb') as f:
    pickle.dump({'age_scaler_min': age_scaler_min, 'age_scaler_max': age_scaler_max,
                 'bmi_scaler_min': bmi_scaler_min, 'bmi_scaler_max': bmi_scaler_max,
                 'ch_scaler_min': ch_scaler_min, 'ch_scaler_max': ch_scaler_max}, f)

#Transform training and test data
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)


# Building Sklearn NN
mlp = MLPRegressor(hidden_layer_sizes=(20,20,5),
                   activation='relu',
                   solver='adam',
                   max_iter=1000,
                   alpha = 0.001,
                   random_state=42)

mlp.fit(X_train_normal, y_train)


# save the model to disk
filename = 'NN_model.sav'
pickle.dump(mlp, open(filename, 'wb'))




