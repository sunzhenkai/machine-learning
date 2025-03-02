"""
房价预测基础数据
"""
import os
import sys
# 添加 root 目录到 path
repo_root_path = str(os.path.abspath('')).replace('/kaggle/comptetitions/HousePricePrediction', '')
sys.path.append(os.path.abspath(repo_root_path))
# 导入特征工程工具库
from internal.feature.feature_engineering import *

# 定义特征列表，包含特征名和类型
feature_set = FeatureSet()
feature_set.features = [
    Feature('Id', FeatureType.ID),
    Feature('MSSubClass', FeatureType.IDENTITY_UNORDERED),
    Feature('MSZoning', FeatureType.IDENTITY_UNORDERED),
    Feature('LotFrontage', FeatureType.VALUE_CONTINUOUS),
    Feature('LotArea', FeatureType.VALUE_CONTINUOUS),
    Feature('Street', FeatureType.IDENTITY_UNORDERED),
    Feature('Alley', FeatureType.IDENTITY_UNORDERED),
    Feature('LotShape', FeatureType.IDENTITY_UNORDERED),
    Feature('LandContour', FeatureType.IDENTITY_UNORDERED),
    Feature('Utilities', FeatureType.IDENTITY_UNORDERED),
    Feature('LotConfig', FeatureType.IDENTITY_UNORDERED),
    Feature('LandSlope', FeatureType.IDENTITY_UNORDERED),
    Feature('Neighborhood', FeatureType.IDENTITY_UNORDERED),
    Feature('Condition1', FeatureType.IDENTITY_UNORDERED),
    Feature('Condition2', FeatureType.IDENTITY_UNORDERED),
    Feature('BldgType', FeatureType.IDENTITY_UNORDERED),
    Feature('HouseStyle', FeatureType.IDENTITY_UNORDERED),
    Feature('OverallQual', FeatureType.IDENTITY_ORDERED),
    Feature('OverallCond', FeatureType.IDENTITY_UNORDERED),
    Feature('YearBuilt', FeatureType.VALUE_DISCRETE),
    Feature('YearRemodAdd', FeatureType.VALUE_DISCRETE),
    Feature('RoofStyle', FeatureType.IDENTITY_UNORDERED),
    Feature('RoofMatl', FeatureType.IDENTITY_UNORDERED),
    Feature('Exterior1st', FeatureType.IDENTITY_UNORDERED),
    Feature('Exterior2nd', FeatureType.IDENTITY_UNORDERED),
    Feature('MasVnrType', FeatureType.IDENTITY_UNORDERED),
    Feature('MasVnrArea', FeatureType.VALUE_CONTINUOUS),
    Feature('ExterQual', FeatureType.IDENTITY_ORDERED),
    Feature('ExterCond', FeatureType.IDENTITY_ORDERED),
    Feature('Foundation', FeatureType.IDENTITY_UNORDERED),
    Feature('BsmtQual', FeatureType.IDENTITY_ORDERED),
    Feature('BsmtCond', FeatureType.IDENTITY_ORDERED),
    Feature('BsmtExposure', FeatureType.IDENTITY_UNORDERED),
    Feature('BsmtFinType1', FeatureType.IDENTITY_ORDERED),
    Feature('BsmtFinSF1', FeatureType.VALUE_CONTINUOUS),
    Feature('BsmtFinType2', FeatureType.IDENTITY_ORDERED),
    Feature('BsmtFinSF2', FeatureType.VALUE_CONTINUOUS),
    Feature('BsmtUnfSF', FeatureType.VALUE_CONTINUOUS),
    Feature('TotalBsmtSF', FeatureType.VALUE_CONTINUOUS),
    Feature('Heating', FeatureType.IDENTITY_UNORDERED),
    Feature('HeatingQC', FeatureType.IDENTITY_ORDERED),
    Feature('CentralAir', FeatureType.IDENTITY_UNORDERED),
    Feature('Electrical', FeatureType.IDENTITY_UNORDERED),
    Feature('1stFlrSF', FeatureType.VALUE_CONTINUOUS),
    Feature('2ndFlrSF', FeatureType.VALUE_CONTINUOUS),
    Feature('LowQualFinSF', FeatureType.VALUE_CONTINUOUS),
    Feature('GrLivArea', FeatureType.VALUE_CONTINUOUS),
    Feature('BsmtFullBath', FeatureType.IDENTITY_UNORDERED),
    Feature('BsmtHalfBath', FeatureType.IDENTITY_UNORDERED),
    Feature('FullBath', FeatureType.IDENTITY_UNORDERED),
    Feature('HalfBath', FeatureType.IDENTITY_UNORDERED),
    Feature('BedroomAbvGr', FeatureType.VALUE_DISCRETE),
    Feature('KitchenAbvGr', FeatureType.VALUE_DISCRETE),
    Feature('KitchenQual', FeatureType.VALUE_DISCRETE),
    Feature('TotRmsAbvGrd', FeatureType.VALUE_DISCRETE),
    Feature('Functional', FeatureType.IDENTITY_UNORDERED),
    Feature('Fireplaces', FeatureType.VALUE_DISCRETE),
    Feature('FireplaceQu', FeatureType.IDENTITY_ORDERED),
    Feature('GarageType', FeatureType.IDENTITY_UNORDERED),
    Feature('GarageYrBlt', FeatureType.VALUE_DISCRETE),
    Feature('GarageFinish', FeatureType.IDENTITY_ORDERED),
    Feature('GarageCars', FeatureType.VALUE_DISCRETE),
    Feature('GarageArea', FeatureType.VALUE_CONTINUOUS),
    Feature('GarageQual', FeatureType.IDENTITY_ORDERED),
    Feature('GarageCond', FeatureType.IDENTITY_ORDERED),
    Feature('PavedDrive', FeatureType.IDENTITY_ORDERED),
    Feature('WoodDeckSF', FeatureType.VALUE_CONTINUOUS),
    Feature('OpenPorchSF', FeatureType.VALUE_CONTINUOUS),
    Feature('EnclosedPorch', FeatureType.VALUE_CONTINUOUS),
    Feature('3SsnPorch', FeatureType.VALUE_CONTINUOUS),
    Feature('ScreenPorch', FeatureType.VALUE_CONTINUOUS),
    Feature('PoolArea', FeatureType.VALUE_CONTINUOUS),
    Feature('PoolQC', FeatureType.IDENTITY_ORDERED),
    Feature('Fence', FeatureType.VALUE_CONTINUOUS),
    Feature('MiscFeature', FeatureType.IDENTITY_UNORDERED),
    Feature('MiscVal', FeatureType.VALUE_CONTINUOUS),
    Feature('MoSold', FeatureType.VALUE_DISCRETE),
    Feature('YrSold', FeatureType.VALUE_DISCRETE),
    Feature('SaleType', FeatureType.IDENTITY_UNORDERED),
    Feature('SaleCondition', FeatureType.IDENTITY_UNORDERED),    
    Feature('SalePrice', FeatureType.LABEL)
]

# 对特征列表做基础的校验（重复特征等）
print(feature_set.check())
# print(feature_set.get_features_by_types(FeatureType.IDENTITY_ORDERED, FeatureType.IDENTITY_UNORDERED))
