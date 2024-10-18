import os
import sys

# 处理工程目录
repo_root_path = str(os.path.abspath('')).replace('/kaggle/comptetitions/HousePricePrediction', '')
# print(os.path.abspath(''), os.path.abspath(repo_root_path))
sys.path.append(os.path.abspath(repo_root_path))  # 工程根目录
sys.path.append(os.path.abspath(os.path.abspath('')))  # 当前文件目录

# 导入依赖库
from internal.feature.feature_engineering import *
from internal.models.tensorflow.tensorflow_linear_regression import TensorflowLinearRegression
from feature_engineering_v1 import dm, train_df


def load():
    pass


def linear_regression_predict():
    lr = TensorflowLinearRegression(dm, epochs=10)
    lr.run()
    pred_y = lr.predict(dm.format_input(train_df))
    print(pred_y)


if __name__ == '__main__':
    linear_regression_predict()
    # pass
