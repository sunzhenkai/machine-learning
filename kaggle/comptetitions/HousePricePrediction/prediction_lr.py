import os
import sys

repo_root_path = str(os.path.abspath('')).replace('/kaggle/comptetitions/HousePricePrediction', '')
# print(os.path.abspath(''), os.path.abspath(repo_root_path))
sys.path.append(os.path.abspath(repo_root_path))  # 工程根目录
sys.path.append(os.path.abspath(os.path.abspath('')))  # 当前文件目录
from internal.feature.feature_engineering import *
from internal.models.pytorch.pytorch_linear_regression import LinearRegression
from feature_engineering_v1 import dm


def load():
    pass


def paddle_lr():
    lr = LinearRegression(dm, epoch=3000)
    lr.run()
    r = lr.predict()
    print(r)


if __name__ == '__main__':
    paddle_lr()
