import torch
import torch.nn as nn
from internal.feature.feature_engineering import DataManager


class LinearRegression(nn.Module):
    def __init__(self, data_manager: DataManager, epoch: int = 3000):
        super(LinearRegression, self).__init__()
        # print(type(data_manager.train_x), type(data_manager.train_y))
        # print('train_y', data_manager.train_y.columns)
        # print(data_manager.train_x.columns, data_manager.train_y)

        self.linear = nn.Linear(len(data_manager.train_x.columns), len(data_manager.train_y.columns))
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.epoch = epoch
        self.data_manager = data_manager

    def forward(self, x):
        return self.linear(x)

    def run(self):
        x_data = torch.from_numpy(self.data_manager.train_x.values).float()
        y_data = torch.from_numpy(self.data_manager.train_y.values).float()
        for i in range(self.epoch):
            y_pred = self.forward(x_data)
            # 计算损失
            loss = self.loss_func(y_pred, y_data)
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # 更新参数
            self.optimizer.step()

    def predict(self):
        x_test = torch.from_numpy(self.data_manager.test_x.values).float()
        return self.forward(x_test)
