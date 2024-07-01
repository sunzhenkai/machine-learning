import os
import sys

repo_root_path = str(os.path.abspath('')).replace('/kaggle/comptetitions/HousePricePrediction', '')
script_path = os.path.abspath(os.path.abspath(''))
print(repo_root_path, script_path)
sys.path.append(os.path.abspath(repo_root_path))
sys.path.append(script_path)

from common import *
import pandas as pd

train_df = pd.read_csv(f'{script_path}/data/train_df_v1.csv')
test_df = pd.read_csv(f'{script_path}/data/test_df_v1.csv')

feature_set_v1 = feature_set.get_features_by_types(FeatureType.IDENTITY_ORDERED, FeatureType.IDENTITY_UNORDERED,
                                                   FeatureType.VALUE_DISCRETE, FeatureType.LABEL)
# print('v1 features:', feature_set_v1.feature_names())

dm = DataManager(feature_set_v1, train_df)
