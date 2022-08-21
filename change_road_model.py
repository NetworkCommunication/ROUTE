"""
该部分是换道模型的代码
实现的是基于GAN的换道模型
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from data_process.config import vehicles_file, road_map_file
from data_process.dataset_choice_road import MatrixRoad
import os

# def dataloader(batch_size=64, train=True):
#     change_road = ChoiceRoad(vehicles_file, road_map_file, train)
#     loader = DataLoader(change_road, batch_size=batch_size, shuffle=True, drop_last=True)
#     return loader
from data_process.road_map import RoadMap



class ChangeRoad(nn.Module):
    def __init__(
            self,
            input_shape
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 3),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        y = self.model(x)
        # y = torch.tensor(y, dtype=torch.float32)
        # y = torch.reshape(y, (-1, 1))
        return y


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()


class ChangeRoadModel:
    def __init__(
            self,
            epoch=51,
            lr=0.0001
    ):
        if os.path.isfile('cnn_change_road_model.pth'):
            self.model = torch.load('cnn_change_road_model.pth')
        else:
            self.model = CNN()
        self.matrix = MatrixRoad()
        self.model.cuda()
        self.epoch = epoch
        self.lr = lr

    def train_model(self):
        loader = dataloader()
        test_loader = dataloader(train=False)
        # summary = SummaryWriter('logs')
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        milestones = [100, 300, 500, 1000, 1500]
        scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
        loss_fun = torch.nn.NLLLoss()
        loss_fun.cuda()
        for i in range(self.epoch):
            iterations = 0
            iterations_t = 0
            average_d = 0
            average_g = 0
            average_t = 0
            right_num = 0
            all_num = 0
            right_num_t = 0
            all_num_t = 0

            for data in loader:
                iterations += 1
                m, l, y, t = data
                y = y[0]
                all_num += len(y)
                m = m.cuda()
                l = l.cuda()
                y = y.cuda()

                y_pre = self.model(l, m)
                optimizer.zero_grad()
                loss = loss_fun(y_pre, y.long())
                loss.backward()

                y_pre = y_pre.detach().cpu()

                y_res = y_pre.argmax(1)

                right_num += torch.sum(y.detach().cpu() == y_res)
                optimizer.step()
                average_g += loss
                # print(t.shape)

            scheduler_g.step()
            # print(iterations)
            print("epoch{}: -----------".format(i))
            print('loss: ', average_g / iterations)
            print('accuracy: ', right_num / all_num)

            if i % 10 == 0:
                torch.save(self.model, 'cnn_change_road_model.pth')

    def predict(self, x, t):
        with torch.no_grad():
            self.model.eval()
            m, l = self.matrix.get_info(t, x)
            m, l = m.cuda(), l.cuda()
            y_pre = self.model(l, m)
            res = road_map.get_many_neighbor(x[:, 0])
            res = torch.tensor(res)
            y_pre = y_pre.detach().cpu()
            y_pre += res * -torch.min(y_pre[0])
            y_res = y_pre.argmax(1)[0].item()
            dir = road_map.next_direction_by_id(x[0][0], y_res)
        return dir


def dataloader(batch_size=1, train=True):
    change_road = MatrixRoad()
    loader = DataLoader(change_road, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_light = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=4),  # (24 - 4) + 1 = 21
            nn.ReLU(True),
            nn.Conv2d(2, 2, kernel_size=4),  # 21 - 3 = 18
            nn.ReLU(True),
            nn.Conv2d(2, 1, kernel_size=4),  # 18 - 3 = 15
            nn.ReLU(True),
            nn.Conv2d(1, 1, kernel_size=3, padding=2, stride=2),  # (15 + 2 * 2 - 3) / 2 + 1 = 9
            nn.ReLU()
        )

        self.cnn_all = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),  # 9 - 2 + 2 = 9
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3),  # 9 - 2 = 7
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3),  # 7 - 2 = 5
            nn.ReLU(True),
            nn.Conv2d(32, 128, kernel_size=3),  # 5 - 2 = 3
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3),  # 3 - 2 = 1
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 200),
            nn.ReLU(True),
            nn.Linear(200, 200),
            nn.ReLU(True),
            nn.Linear(200, 200),
            nn.ReLU(True),
            nn.Linear(200, 24),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, light, normal):
        light = torch.reshape(light, (light.shape[0], 1, light.shape[1], light.shape[2]))
        t = self.cnn_light(light)
        o = torch.cat(
            (
                t, normal
            ),
            dim=1
        )
        o = self.cnn_all(o)
        o = torch.reshape(o, (o.shape[0], -1))
        o = self.fc(o)
        return o


if __name__ == '__main__':
    road_map = RoadMap()
    change_road_model = ChangeRoadModel()
    # change_road_model.train_model()
    res = change_road_model.predict(np.array([[15, 1]]), [0, ])
    print(res)
    # print(road_map.next_direction_by_id(12, 3))
    print(road_map.get_neighbor(15))
