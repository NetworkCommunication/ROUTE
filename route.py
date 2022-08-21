import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from change_road_model import ChangeRoad
from config import grid_temporal_cache
from coordinate_conversion import CoordinateConversion
from datasets import VehiclesData
from grid_map import get_grid_from_map

from model import Generator, Discriminator
import numpy as np
import random
import time

from road_map import RoadMap
from tools import get_road_name
from traffic_light import TrafficLightSet
from vehicle_state import VehicleState

N_GRID = 39
T = 30
WINDOW = 5
min_loss = 100

MIN_DISTANCE = 5

INTERSECTION_TIME = 2  # 车辆在交叉路口停留的时间，SUMO中都是1-3S
LENGTH = 500

grid_cache = {}
grid_test_cache = {}


def dataloader(batch_size=1, train=True):
    label = '训练' if train else '测试'
    print("====================================加载{}数据====================================".format(label))
    grip_data = VehiclesData(train=train)
    loader = DataLoader(grip_data, batch_size=batch_size)
    print("加载完毕，{}数据数量为{}".format(label, len(grip_data)))
    return loader


def del_tensor(tensors: torch.tensor, index: np.ndarray):
    # 删除tensor张量中的行
    ten = tensors.detach().cpu().numpy()
    ten = np.delete(ten, index, axis=0)
    return torch.tensor(ten).cuda()


def save(grid: np.ndarray, vid, cache=None):
    global grid_cache
    global grid_test_cache
    # print(grid)
    # 将每辆车的网格信息保存起来
    grid_for_v = {}
    grid = grid.tolist()
    for key, _ in enumerate(vid):
        vehicle = vid[key]
        grid_for_v[vehicle] = grid[key]
    # with open(grid_temporal_cache, 'w') as f:
    #     json.dump(grid_for_v, f)
    if cache is None:
        grid_cache = grid_for_v
    else:
        grid_test_cache = grid_for_v


def update_grid(grid_now: list, vid, cache=None):
    # 更新网格信息
    global grid_cache
    global grid_test_cache
    if cache is None:
        grid = grid_cache
    else:
        grid = grid_test_cache
    # with open(grid_temporal_cache, 'r') as f:
    #     grid = json.load(f)
    for key, _ in enumerate(vid):
        vehicle = vid[key]
        # print(vehicle)
        grid_history = grid[vehicle]
        if grid_now[key] != 0:
            # gird_history[0], gird_history[1], gird_history[2], grid_history[3], grid_history[4] = \
            #     gird_history[1], gird_history[2], grid_history[3], grid_history[5], grid_now[key]
            grid_history[:-1] = grid_history[1:]
            grid_history[-1] = grid_now[key]
        else:
            # gird_history[0], gird_history[1], gird_history[2], grid_history[3], grid_history[4] = \
                # gird_history[1], gird_history[2], grid_history[3], grid_history[4], np.zeros((N_GRID, 9)).tolist()
            grid_history[:-1] = grid_history[1:]
            grid_history[-1] = np.zeros((N_GRID, 9)).tolist()

        grid[vehicle] = grid_history
    if cache is None:
        grid_cache = grid
    else:
        grid_test_cache = grid


def update_intersection_vehicle(intersection_vehicle: dict, cache=None):
    # 更新交叉路口中的车辆，t-1 更新
    vids = []
    out_inter = {}
    need_del = []
    for k in intersection_vehicle:
        intersection_vehicle[k]['t'] -= 1

        # 如果车辆到了出交叉路口的时候， 则将其从中删除，同时返回删除的车辆信息
        if intersection_vehicle[k]['t'] == 0:
            out_inter[k] = intersection_vehicle[k]
            need_del.append(k)
        else:
            vids.append(k)

    for k in need_del:
        intersection_vehicle.pop(k)

    update_grid([0 for _ in vids], vids, cache)
    return intersection_vehicle, out_inter


def load_grid(vid, n_grid, cache=None):
    # 加载网格信息，根据vid的大小返回网格信息
    grid = np.zeros((len(vid), WINDOW, n_grid, 9))

    if cache is None:
        grid_history = grid_cache
    else:
        grid_history = grid_test_cache
    for i, _ in enumerate(vid):
        vehicle = vid[i]
        # grid_s = np.zeros((5, n_grid, 9))
        grid_v = np.array(grid_history[vehicle])
        # print(grid_v.shape)
        # for k in range(grid_v.shape[0]):
        #     for j in range(grid_v.shape[1]):
        #         grid_s[k, j] = np.array(grid_v[k, j])
        # grid_v = np.reshape(grid_v, (grid_v.shape[0], grid_v.shape[1], -1))
        # grid_s = grid_v
        grid[i] = grid_v

    return grid


def find_near_intersection_vehicle(state: list, vid: list):
    """
    获取即将到达路口或者已经到达路口的车辆
    :param state: [car_num, 3]  [lane_number, position, speed]
    :param vid: [car_num]
    :return:
    """

    # 当车辆下一秒就驶出路口时，判断车辆处于路口中
    near_list = list(filter(
        lambda v: LENGTH - state[vid.index(v)][1] - state[vid.index(v)][2] <= MIN_DISTANCE, vid))
    return near_list


# best n_spatial_layer=3, n_temporal_layer=3, emb_dim=32, n_head=8
class ROUTE(nn.Module):
    def __init__(self, n_input=9, output_dim=2, t=5, n_grid=39, n_spatial_layer=3, n_temporal_layer=3, emb_dim=64,
                 n_head=16, drop_rate=0.1,
                 learning_rate=0.0001, epochs=500):
        super().__init__()

        # 初始化generator和discriminator
        self.g_net = Generator(n_input, output_dim, t, n_grid, 'reg', n_spatial_layer, n_temporal_layer, emb_dim,
                               n_head, drop_rate)
        self.d_net = Discriminator(n_input, output_dim, t, n_grid, n_spatial_layer, n_temporal_layer, emb_dim, n_head,
                                   drop_rate)
        self.classifier = Generator(n_input, output_dim, t, n_grid, 'classifier', n_spatial_layer, n_temporal_layer,
                                    emb_dim, n_head, drop_rate)

        self.g_net.cuda()
        self.d_net.cuda()
        self.classifier.cuda()

        self.g_net = nn.DataParallel(self.g_net)
        self.d_net = nn.DataParallel(self.d_net)
        self.classifier = nn.DataParallel(self.classifier)

        self.opt_g = torch.optim.Adam(
            self.g_net.parameters(),
            lr=learning_rate
        )
        self.opt_d = torch.optim.Adam(
            self.d_net.parameters(),
            lr=learning_rate
        )
        self.opt_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=learning_rate
        )

        self.milestones = [50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000]
        self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(self.opt_g, milestones=self.milestones, gamma=0.5)
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR(self.opt_d, milestones=self.milestones, gamma=0.5)
        self.scheduler_c = torch.optim.lr_scheduler.MultiStepLR(self.opt_c, milestones=self.milestones, gamma=0.5)

        self.criterion_minimize = nn.MSELoss().cuda()
        self.criterion = nn.BCELoss().cuda()
        self.criterion_classifier = nn.NLLLoss().cuda()

        self.vehicle_state = VehicleState()  # 获取车辆的状态真实值
        self.tls = TrafficLightSet()
        self.change_road = torch.load('change_road_model.pth')  # 换道模型
        self.train_data = dataloader(train=True)
        self.test_data = dataloader(train=False)
        self.coordinate_conversion = CoordinateConversion()
        self.road_map = RoadMap()

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.t_length = 2

    def make_grid(self, output: np.ndarray, vid: np.ndarray, vehicle_states: np.ndarray, t: int):
        # 将数据转化为路网的格式
        # output_type : [n_car, 3]  [lane_num, lane_position, speed]
        # vid : vehicle id list
        # vehicle_states : x, y, road_id, lane_num, position_lane, a, v_s, h_s
        assert output.shape[0] == vid.shape[0] and vid.shape[0] == vehicle_states.shape[0], "vehicle number need be " \
                                                                                            "consistent "
        # [position_lane, lateral_position, s, d, angle, lane_number, light, g_s, road_id, g_d]
        road_net = {}
        road_list = vehicle_states[:, 2]
        road_set = list(set(road_list))  # 去除重复的id
        # print(road_set)
        road_name_set = [self.road_map.get_road_name(int(rid)) for rid in road_set]  # 道路名集合

        # 初始化字典
        for rname in road_name_set:
            road_net[rname] = {
                "vehicles": {},
                "road_id": rname
            }

        for key, _ in enumerate(vid):
            # 车辆id
            vehicle = vid[key]
            speed = output[key, 2]
            road_id = vehicle_states[key, 2]
            road_name = self.road_map.get_road_name(int(road_id))
            lane_num = output[key, 0]
            position_in_lane = output[key, 1]
            path = self.vehicle_state.get_vehicle_trip(vehicle)
            vinfo = {
                "id": vehicle,
                "info": {
                    'position': [0, 0],
                    "position_in_lane": position_in_lane,
                    "speed": speed,
                    "lane": "{}_{}".format(road_name, int(lane_num)),
                    "road": road_name,
                    "length": 5,
                    "width": 1.8,
                    "accelerate": 0,
                    "route": path,
                    "lateral_position": 0,
                    "lateral_speed": 0,
                    "angle": 0
                }
            }
            road_net[road_name]['vehicles'][vehicle] = vinfo

        # print(road_net)
        grip_now, vehicles_id = get_grid_from_map(t, {"vehicles": road_net})
        grip_now = torch.tensor(grip_now)
        grids = torch.zeros((grip_now.shape[2] - 1, grip_now.shape[0],
                             grip_now.shape[1]))
        t = torch.permute(grip_now, (2, 0, 1))
        grids[0], grids[1:] = t[0], t[2:]  # 当前时间的路网车辆网格信息
        grids = torch.permute(grids, (1, 2, 0))
        return np.array(grids).tolist(), vehicles_id

    def train_model(self):
        """
        训练模型，训练模型的过程首先将一秒一秒的训练
        如果预测下一秒的误差太大会很影响最终模型的结果
        :return:
        """
        summary = SummaryWriter('route_log')
        min_position = 10
        min_speed = 10

        # 首先进行100次预训练，只训练预测一秒
        for epoch in range(200):
            loss_g = 0
            loss_d = 0
            car_num = 0
            mse_loss = 0
            num_mse = 0
            num = 0
            num_g = 1
            acc_num = 0
            acc = 0
            speed_loss = 0
            position_loss = 0
            all = 0
            for data in self.train_data:
                # 由于每一秒的车辆数据太多,因此将数据划分为几次训练,而不是一次训练完成所有的工作
                grid, vid, t_0 = data  # [1, n_car, n_grid, input_shape]
                vid = [v[0] for v in vid]
                all += len(vid)
                # 训练模型
                _input = torch.zeros((WINDOW, grid.shape[1], grid.shape[2],
                                      grid.shape[3]))  # [window, n_car, n_grid, input_shape]
                _input[-1] = grid[0]
                # _input[0], _input[1], _input[2], _input[3], _input[4] = grid[0], grid[0], grid[0], grid[0], \
                #                                                         grid[0]
                _input = torch.permute(_input, (1, 0, 2, 3))
                _input = _input.cuda()

                # x, y, road_id, lane_num, position_lane, a, v_s, h_s
                real = self.vehicle_state.get_some_vehicles_state(int(t_0), vid)
                real = np.array(real)

                # 处理分三种情况
                # 1. 到达路口即为终点  删除该车辆，不放入预测
                # 2. 到达路口需要换道  将车辆放入换道集合中，并计算换道需要的时间，没个循环让这个时间减一
                # 3. 预测数据显示到达路口其实没有到  不进行处理
                # 那么就需要插下一秒的真实数据的情况，如果真实数据的情况是车辆已经不再当前RSG，那么不在预测下一秒的序列中存在
                # print(_input.shape)
                error_list = np.where(real[:, 2] < 0)  # 不需要放入神经网络训练的车辆
                real = np.delete(real, error_list, axis=0)
                _input = del_tensor(_input, error_list)

                position = np.zeros((_input.shape[0], 3))
                position_real = np.zeros((_input.shape[0], 3))

                fake_label = torch.zeros((_input.shape[0], 1)).cuda()
                real_label = torch.ones((_input.shape[0], 1)).cuda()

                # generator 的输入格式 [n_car, t, n_grid, input_shape]
                fake_o = self.g_net(_input)
                fake_c = self.classifier(_input)

                fake_t = torch.zeros((fake_o.shape[0], 1 + fake_o.shape[1])).cuda()
                fake_t[:, 0], fake_t[:, 1:] = fake_c.argmax(1), fake_o

                real_t = np.zeros((fake_t.shape[0], fake_t.shape[1]))
                real_t[:, :] = real[:, 3:6]  # x, y
                real_t = torch.tensor(real_t, dtype=torch.float32).cuda()

                real_c = real_t.detach().cpu().numpy()[:, 0]
                real_c = torch.tensor(real_c).cuda()

                real_o = real_t.detach().cpu().numpy()[:, 1:]
                real_o = torch.tensor(real_o).cuda()

                # 训练discriminator
                self.opt_d.zero_grad()
                d_real = self.d_net(_input, real_o.detach())
                d_fake = self.d_net(_input, fake_o.detach())

                # print(d_real, real_label)
                d_real_loss = self.criterion(d_real, real_label)
                d_fake_loss = self.criterion(d_fake, fake_label)

                d_loss = (d_real_loss + d_fake_loss)
                d_loss.backward()
                self.opt_d.step()
                loss_d += d_loss

                position[:, 2] = real[:, 2]
                position[:, :2] = fake_t.detach().cpu().numpy()[:, :2]
                position_real[:, 2] = real[:, 2]
                position_real[:, :2] = real_t.detach().cpu().numpy()[:, :2]

                # [n_car, 2]
                position = np.array(self.coordinate_conversion.converse_some_conversion(position))
                position_real = np.array(self.coordinate_conversion.converse_some_conversion(position_real))
                position_loss += np.mean(
                    np.sqrt(
                        (position[:, 0] - position_real[:, 0]) ** 2 + (position[:, 1] - position_real[:, 1]) ** 2))
                # print(position_loss)
                speed = fake_o.detach().cpu().numpy()[:, -1]
                speed_real = real_t.detach().cpu().numpy()[:, -1]
                speed_loss += np.mean(
                    np.abs(speed - speed_real)
                )

                # 平均每训练三次判别器，训练一次生成器
                self.opt_g.zero_grad()
                d_fake_1 = self.d_net(_input, fake_o)
                g_ad_loss = self.criterion(d_fake_1, real_label)
                g_l2_loss = self.criterion_minimize(fake_o, real_o)
                g_loss = g_l2_loss * 0.5 + g_ad_loss
                g_loss.backward()
                self.opt_g.step()
                loss_g += g_loss
                num_g += 1

                self.opt_c.zero_grad()
                nl_loss = self.criterion_classifier(fake_c, real_c.long())
                nl_loss.backward()
                self.opt_c.step()

                acc += torch.sum(fake_c.argmax(1) == real_c)
                acc_num += fake_c.shape[0]

                # loss_d += d_fake_loss + d_real_loss

                car_num += 1

            self.test(epoch, summary)
            # print(all)
            num += 1
            # print(car_num)

            # self.scheduler_d.step()
            self.scheduler_g.step()
            self.scheduler_c.step()

            print("epoch {} -------------------".format(epoch))
            print("classifier accuracy is {}".format(acc / acc_num))
            # print("learning rate is {}".format(self.scheduler_g.get_last_lr()))
            # print("mse loss is {}".format(mse_loss / num_mse))
            print("the average loss is {}".format(position_loss / car_num))
            print("loss_d is {}".format(loss_d / num_g))
            print("loss_g is {}".format(loss_g / num_g))
            print("speed loss is {}".format(speed_loss / car_num))
            print()

            # if position_loss / car_num < min_position:
            #     min_position = position_loss / car_num
            #     min_speed = speed_loss / car_num

            summary.add_scalar("position_loss", position_loss / car_num, epoch)
            summary.add_scalar("loss_d", loss_d / num, epoch)
            summary.add_scalar("loss_g", loss_g / num_g, epoch)
            # summary.add_scalar("mse_loss", mse_loss / num_mse, epoch)
            summary.add_scalar("acc", acc / acc_num, epoch)
            summary.add_scalar("speed", speed_loss / car_num, epoch)

        print(min_position, min_speed)
        # torch.save(self.d_net, 'discriminator.pth')
        # torch.save(self.g_net, 'generator.pth')
        # torch.save(self.classifier, 'classifier.pth')
        # self.train_multi_second()
        summary.close()
        # for data in self.test_data:
        #     print(data)

    def train_multi_second(self):
        """
        只训练5s时间长度的模型，因为时间窗口长度为5s
        :return:
        """
        length = WINDOW  # 训练窗口的大小

        for epoch in range(3000):
            num = [0 for _ in range(length)]
            mse_loss = [0 for _ in range(length)]
            acc = [0 for _ in range(length)]
            speed_loss = [0 for _ in range(length)]
            position_loss = [0 for _ in range(length)]
            loss_d = [0 for _ in range(length)]
            n = 0
            for data in self.train_data:
                n += 1
                grid, vid, t_0 = data
                vid = [v[0] for v in vid]
                vid = np.array(vid)
                vehicles_in_intersection = {}  # 保存换道中的车辆信息，vid、交叉路口中停留的时间

                for i in range(length):
                    # 如果是一开始，那么需要加载grid的数据， 否则需要从文件中加载grid数据
                    _input = torch.zeros((WINDOW, grid.shape[1], grid.shape[2],
                                          grid.shape[3]))  # [window, n_car, n_grid, input_shape]

                    if i == 0:
                        # _input[0], _input[1], _input[2], _input[3], _input[4] = grid[0], grid[0], grid[0], grid[0], \
                        #                                                         grid[0]
                        _input[-1] = grid[0]
                        _input = torch.permute(_input, (1, 0, 2, 3))
                        save(np.array(_input), vid)
                    else:  # 从文件中加载grid
                        grid = load_grid(vid, N_GRID)
                        _input = torch.tensor(grid, dtype=torch.float32)

                    _input = _input.cuda()

                    real = self.vehicle_state.get_some_vehicles_state(int(t_0) + i, vid)
                    real = np.array(real)

                    # 处理在下一秒即将进入交叉路口或者到达目的地的车辆
                    intersection_list = np.where(real[:, 2] == -1)[0]  # 即将进入交叉路口的车辆
                    intersection_list = [vid[i] for i in intersection_list]  # 保存即将进入交叉路口的车辆的ID

                    # 将需要进入交叉路口的车辆信息保存
                    for k, _ in enumerate(intersection_list):
                        vehicle = intersection_list[k]
                        vehicles_in_intersection[vehicle] = {
                            't': 2,
                            'info': real[k]
                        }

                    # 准备数据的工作完成之后，需要进行模型的训练部分
                    # 删除需要删除的部分
                    # 先删除消失的，删除的先后顺序不能乱，不然就会删除数据
                    need_del_list = np.where(real[:, 2] == -2)[0]  # 已经到达目的地的车辆
                    vid = np.delete(vid, need_del_list)
                    _input = del_tensor(_input, need_del_list)
                    real = np.delete(real, np.where(real[:, 2] == -2), axis=0)

                    # 再删除进入交叉路口的
                    vid = np.delete(vid, np.where(real[:, 2] == -1))
                    _input = del_tensor(_input, np.where(real[:, 2] == -1))
                    real = np.delete(real, np.where(real[:, 2] == -1), axis=0)

                    # generator 的输入格式 [n_car, t, n_grid, input_shape]
                    fake_o = self.g_net(_input)
                    fake_o[fake_o[:, 0] > 499, 0] = 499
                    fake_c = self.classifier(_input)

                    fake_t = torch.zeros((fake_o.shape[0], 1 + fake_o.shape[1])).cuda()
                    fake_t[:, 0], fake_t[:, 1:] = fake_c.argmax(1), fake_o

                    real_t = real[:, 3:6]
                    real_t = torch.tensor(real_t, dtype=torch.float32).cuda()

                    real_o = real_t.detach().cpu().numpy()[:, 1:]
                    real_o = torch.tensor(real_o).cuda()

                    real_c = real_t.detach().cpu().numpy()[:, 0]
                    real_c = torch.tensor(real_c).cuda()

                    position = np.zeros((_input.shape[0], 3))
                    position_real = np.zeros((_input.shape[0], 3))

                    position[:, 2] = real[:, 2]
                    position[:, :2] = fake_t.detach().cpu().numpy()[:, :2]
                    position_real[:, 2] = real[:, 2]
                    position_real[:, :2] = real_t.detach().cpu().numpy()[:, :2]

                    # [n_car, 2]
                    position = np.array(self.coordinate_conversion.converse_some_conversion(position))
                    position_real = np.array(self.coordinate_conversion.converse_some_conversion(position_real))
                    position_loss[i] += np.mean(
                        np.sqrt(
                            (position[:, 0] - position_real[:, 0]) ** 2 + (position[:, 1] - position_real[:, 1]) ** 2
                        )
                    )
                    num[i] += 1

                    speed = fake_o.detach().cpu().numpy()[:, -1]
                    speed_real = real_t.detach().cpu().numpy()[:, -1]
                    speed_loss[i] += np.mean(
                        np.abs(speed - speed_real)
                    )
                    fake_label = torch.zeros((_input.shape[0], 1)).cuda()
                    real_label = torch.ones((_input.shape[0], 1)).cuda()

                    # 更新
                    self.opt_d.zero_grad()
                    d_real = self.d_net(_input, real_o.detach())
                    d_fake = self.d_net(_input, fake_o.detach())

                    # print(d_real.shape, real_label.shape)
                    d_real_loss = self.criterion(d_real, real_label)
                    d_fake_loss = self.criterion(d_fake, fake_label)

                    d_loss = (d_real_loss + d_fake_loss)
                    d_loss.backward()
                    self.opt_d.step()
                    loss_d[i] += d_loss.view(-1).item()

                    self.opt_g.zero_grad()
                    d_fake_1 = self.d_net(_input, fake_o)
                    g_ad_loss = self.criterion(d_fake_1, real_label)
                    g_l2_loss = self.criterion_minimize(fake_o, real_o)
                    g_loss = (g_l2_loss * 0.01 + g_ad_loss)
                    g_loss.backward()
                    self.opt_g.step()
                    mse_loss[i] += g_loss.view(-1).item()

                    # 更新
                    nl_loss = self.criterion_classifier(fake_c, real_c.long())
                    self.opt_c.zero_grad()
                    nl_loss.backward()
                    self.opt_c.step()

                    acc[i] += (torch.sum(fake_c.argmax(1) == real_c) / fake_c.shape[0]).view(-1).item()

                    # 预测完之后，需要更新数据
                    # 将交叉路口中的车辆停留时间数减一
                    # 更新新的网格信息
                    vehicles_in_intersection, out_inter = update_intersection_vehicle(vehicles_in_intersection)
                    # x, y, road_id, lane_num, position_lane, a, v_s, h_s
                    output = fake_t.detach().cpu().numpy()  # 直接使用真实的数据
                    # 将当前出交叉路口的车辆放入id列表中
                    for vehicle in out_inter:
                        # 当前需要准备的数据是t_0 + i时间的数据
                        # 50s 的数据 51s
                        v_s = self.vehicle_state.get_one_vehicle_state(int(t_0) + i, vehicle)
                        # 如果当前车辆还在交叉路口中，那么继续讲车辆放入到字典中
                        if type(v_s[2]) == str:
                            vehicles_in_intersection[vehicle] = {
                                't': 1,
                                'info': v_s
                            }
                            continue
                        l = np.array(v_s)
                        l = np.reshape(l, (1, -1))
                        l_t = l[:, 3:6]
                        vid = np.append(vid, vehicle)

                        real = np.append(real, l, axis=0)
                        output = np.append(output, l_t, axis=0)

                    # 更新网格数据
                    grip_now, vehicles_id = self.make_grid(output, vid, real, int(t_0) + i)
                    update_grid(grip_now, vehicles_id)

            acc = [acc[k] / num[k] for k, _ in enumerate(acc)]
            position_loss = [position_loss[k] / num[k] for k, _ in enumerate(position_loss)]
            speed_loss = [speed_loss[k] / num[k] for k, _ in enumerate(speed_loss)]
            mse_loss = [mse_loss[k] / num[k] for k, _ in enumerate(mse_loss)]
            loss_d = [loss_d[k] / num[k] for k, _ in enumerate(loss_d)]

            self.scheduler_g.step()
            self.scheduler_c.step()
            self.scheduler_d.step()

            print("epoch {} -------------------".format(epoch))
            print("classifier accuracy is ", acc)
            print("loss g is ", mse_loss)
            print("loss d is ", loss_d)
            print("the average loss is ", position_loss)
            print("speed loss is ", speed_loss)
            with open('train_log_new_{}.json'.format(WINDOW), 'w') as f:
                json.dump({'epoch': epoch, 'acc': acc, 'position_loss': position_loss, 'speed_loss': speed_loss}, f)

            if epoch > 100:
                self.multi_test(epoch)
            
            # if epoch % 100 == 0:

            # self.test(epoch, None)
            # if epoch % 30 == 0:
            #     torch.save(self.generator, 'generator.pth')
            #     torch.save(self.classifier, 'classifier.pth')

    def multi_test(self, epoch=0):
        """
        多秒的测试模型，也可以作为最后的预测部分
        :return:
        """
        length = 30  # 测试时间长度的大小
        global min_loss
        classifier = self.classifier
        generator = self.g_net
        # classifier = torch.load('./classifier_new_5.pth')
        # generator = torch.load('./generator_new_5.pth')

        num = [0 for _ in range(length)]
        acc = [0 for _ in range(length)]
        speed_loss = [0 for _ in range(length)]
        position_loss = [0 for _ in range(length)]

        n = 0

        for data in self.test_data:
            # s_y, s, d, angle, s_lane, light, g_s, road_id, g_d
            n += 1
            # if n != 5 and n != 6:
                # continue
            grid, vid, t_0 = data
            vid = [v[0] for v in vid]
            vid = np.array(vid)
            vehicles_in_intersection = {}  # 保存换道中的车辆信息，vid、交叉路口中停留的时间
            # x, y, road_id, lane_num, position_in_lane, v_s, a, h_s
            initial_info = self.vehicle_state.get_some_vehicles_state(int(t_0) - 1, vid.tolist())  # 获取初始车辆的信息
            road_list = {}  # 记录每个车辆当前的所在RSG编号
            direction_list = {}  # 保存当前车辆的下一次转向方向
            now_state = np.array(initial_info)[:, 3:6]  # 保存每秒的车辆状态
            light_state = {}  # 保存当前红路灯状态
            speed_list = {}
            for key, _ in enumerate(initial_info):
                vehicle = vid[key]
                info = initial_info[key]
                road_list[vehicle] = info[2]

            classifier.eval()
            generator.eval()
            with torch.no_grad():
                for i in range(length):
                    # 如果是一开始，那么需要加载grid的数据， 否则需要从文件中加载grid数据
                    _input = torch.zeros((WINDOW, grid.shape[1], grid.shape[2],
                                          grid.shape[3]))  # [window, n_car, n_grid, input_shape]

                    if i == 0:
                        _input[-1] = grid[0]
                        _input = torch.permute(_input, (1, 0, 2, 3))
                        save(np.array(_input), vid, cache='test')
                    else:  # 从文件中加载grid
                        grid = load_grid(vid, N_GRID, cache='test')
                        _input = torch.tensor(grid, dtype=torch.float32)

                    _input = _input.cuda()

                    error = 0
                    for key, _ in enumerate(vid):
                        vehicle = vid[key]
                        # a = _input[key]
                        # v_g = a[68888888888]
                        # v_t = v_g[13]
                        target_road = self.vehicle_state.get_vehicle_trip(vehicle)[-1]
                        target_road = self.road_map.get_road_id(target_road)
                        current_road = road_list[vehicle]
                        if current_road == -1:
                            error += 1
                        aim_info = torch.tensor(np.array([[current_road, target_road]]),
                                                dtype=torch.float32)
                        speed_list[vehicle] = now_state[key, -1
                        ]
                        aim_info = aim_info.cuda()
                        a_s = self.change_road(aim_info)
                        angle = a_s.argmax(1)[0].item()
                        lane, junction = self.road_map.next_road(int(road_list[vehicle]), angle)
                        t = int(t_0) + i - 1
                        try:
                            light_state[vehicle] = self.tls.get_current_state(t, junction.split('_')[0],
                                                                              junction) if target_road != current_road else 2
                            if self.tls.get_current_state(t + 1, junction.split('_')[0],
                                                          junction) == 1 and light_state[vehicle] == 0:
                                light_state[vehicle] = 1
                            direction_list[vehicle] = angle
                        except AttributeError:
                            a_s = a_s.detach().cpu().numpy()[0]
                            a_s[a_s.argmax()] = -999
                            angle = a_s.argmax()

                            lane, junction = self.road_map.next_road(int(road_list[vehicle]), angle)
                            if junction is not None:
                                light_state[vehicle] = self.tls.get_current_state(t, junction.split('_')[0],
                                                                                  junction) if target_road != current_road else 2
                                direction_list[vehicle] = angle
                            else:
                                a_s = a_s.detach().cpu().numpy()[0]
                                a_s[a_s.argmax()] = -999
                                angle = a_s.argmax()
                                print(lane, angle, junction, current_road, road_list[vehicle])
                                print(vehicle)
                                lane, junction = self.road_map.next_road(int(road_list[vehicle]), angle)
                                light_state[vehicle] = self.tls.get_current_state(t, junction.split('_')[0],
                                                                                  junction) if target_road != current_road else 2
                                direction_list[vehicle] = angle
                        if get_road_name(lane) not in self.vehicle_state.get_vehicle_trip(vehicle):
                            error += 1

                    # print("error number is {}".format(error))
                    near_list = find_near_intersection_vehicle(now_state, vid.tolist())
                    need_del = []
                    need_out = []
                    # print(int(t_0) + i)
                    real = self.vehicle_state.get_some_vehicles_state(int(t_0) + i, vid)
                    real = np.array(real)

                    for vehicle in near_list:
                        if light_state[vehicle] == 2:
                            need_del.append(vehicle)
                        elif light_state[vehicle] == 1:
                            need_out.append(vehicle)

                    intersection_list = np.where(real[:, 2] == -1)[0]  # 即将进入交叉路口的车辆
                    intersection_list = [vid[i] for i in intersection_list]  # 保存即将进入交叉路口的车辆的ID

                    # 将需要进入交叉路口的车辆信息保存
                    for k, _ in enumerate(need_out):
                        vehicle = need_out[k]
                        if speed_list[vehicle] == 0:
                            t_in_intersection = 3
                        else:
                            t_in_intersection = 2
                        vehicles_in_intersection[vehicle] = {
                            't': t_in_intersection,
                        }
                        road_list[vehicle] = self.road_map.get_road_id(
                            self.road_map.get_nex_road(road_list[vehicle], direction_list[vehicle])
                        )

                    need_del_list = np.where(real[:, 2] == -2)[0]  # 已经到达目的地的车辆
                    # vid = np.delete(vid, need_del_list)
                    # _input = del_tensor(_input, need_del_list)
                    # real = np.delete(real, np.where(real[:, 2] == -2), axis=0)
                    #
                    # # 再删除进入交叉路口的
                    # vid = np.delete(vid, np.where(real[:, 2] == -1))
                    # _input = del_tensor(_input, np.where(real[:, 2] == -1))
                    # real = np.delete(real, np.where(real[:, 2] == -1), axis=0)
                    need_del_list = []
                    need_out_list = []
                    for val in need_del:
                        need_del_list.append(np.argwhere(vid == val)[0])
                    if len(need_del_list) != 0:
                        need_del_list = np.array(need_del_list)
                        vid = np.delete(vid, need_del_list)
                        _input = del_tensor(_input, need_del_list)
                        real = np.delete(real, need_del_list, axis=0)

                    # 再删除进入交叉路口的
                    for val in need_out:
                        need_out_list.append(np.argwhere(vid == val)[0])
                    if len(need_out_list) != 0:
                        need_out_list = np.array(need_out_list)
                        vid = np.delete(vid, need_out_list)
                        _input = del_tensor(_input, need_out_list)
                        real = np.delete(real, need_out_list, axis=0)

                    # generator 的输入格式 [n_car, t, n_grid, input_shape]
                    fake_o = generator(_input)
                    fake_o[fake_o[:, 0] > 499, 0] = 499
                    fake_c = classifier(_input)

                    fake_t = torch.zeros((fake_o.shape[0], 1 + fake_o.shape[1])).cuda()
                    fake_t[:, 0], fake_t[:, 1:] = fake_c.argmax(1), fake_o

                    real_t = real[:, 3:6]
                    real_t = torch.tensor(real_t, dtype=torch.float32).cuda()

                    real_o = real_t.detach().cpu().numpy()[:, 1:]
                    real_o = torch.tensor(real_o).cuda()

                    real_c = real_t.detach().cpu().numpy()[:, 0]
                    real_c = torch.tensor(real_c).cuda()

                    position = np.zeros((_input.shape[0], 3))
                    position_real = np.zeros((_input.shape[0], 3))

                    position[:, 2] = real[:, 2]
                    position[:, :2] = fake_t.detach().cpu().numpy()[:, :2]
                    position_real[:, 2] = real[:, 2]
                    position_real[:, :2] = real_t.detach().cpu().numpy()[:, :2]

                    # [lane_name, position_in_lane, road_id]
                    error_list = np.where(position_real[:, 2] < 0)[0]
                    for val in error_list:
                        if position[val, 1] > 450:
                            position_real[val, 1] = 500
                        elif position[val, 1] < 50:
                            position_real[val, 1] = 0
                        position_real[val, 2] = position[val, 2]

                    # [n_car, 2]
                    position = np.array(self.coordinate_conversion.converse_some_conversion(position))
                    position_real = np.array(self.coordinate_conversion.converse_some_conversion(position_real))
                    position_loss[i] += np.mean(
                        np.sqrt(
                            (position[:, 0] - position_real[:, 0]) ** 2 + (position[:, 1] - position_real[:, 1]) ** 2
                        )
                    )
                    num[i] += 1

                    speed = fake_o.detach().cpu().numpy()[:, -1]
                    speed_real = real_t.detach().cpu().numpy()[:, -1]
                    speed_loss[i] += np.mean(
                        np.abs(speed - speed_real)
                    )
                    # if i == length - 1:
                    #     self.statistic_position(position_real, position)
                    #     self.statistic_speed(speed_real, speed)

                    acc[i] += (torch.sum(fake_c.argmax(1) == real_c) / fake_c.shape[0]).view(-1).item()

                    # 预测完之后，需要更新数据
                    # 将交叉路口中的车辆停留时间数减一
                    # 更新新的网格信息
                    vehicles_in_intersection, out_inter = update_intersection_vehicle(vehicles_in_intersection, cache='test')
                    # x, y, road_id, lane_num, position_lane, a, v_s, h_s
                    output = fake_t.detach().cpu().numpy()  # 直接使用真实的数据
                    real_fake = np.zeros((real.shape[0], real.shape[1]))

                    # # 将real_fake内容补齐，real_fake只用到它的道路id
                    real_fake = real
                    for vehicle in vid:
                        index = np.argwhere(vid == vehicle)
                        real_fake[index, 2] = road_list[vehicle]

                    # 将当前出交叉路口的车辆放入id列表中
                    for vehicle in out_inter:
                        # 当前需要准备的数据是t_0 + i时间的数据
                        # 50s 的数据 51s
                        # 如果当前车辆还在交叉路口中，那么继续讲车辆放入到字典中
                        l = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                        l = np.reshape(l, (1, -1))
                        l_t = l[:, 3:6]
                        vid = np.append(vid, vehicle)
                        l[:, 2] = road_list[vehicle]
                        real_fake = np.append(real_fake, l, axis=0)
                        output = np.append(output, l_t, axis=0)

                    # 更新网格数据
                    grip_now, vehicles_id = self.make_grid(output, vid, real_fake, int(t_0) + i)
                    update_grid(grip_now, vehicles_id, cache="test")

                    vid = np.array(vehicles_id)
                    a = np.array(grip_now)
                    a_t = a[:, N_GRID // 2]
                    now_state = np.zeros((vid.shape[0], 3))
                    now_state[:, 1] = a_t[:, 0]
                    now_state[:, 0] = a_t[:, 4]
                    now_state[:, 2] = a_t[:, 1]

        acc = [acc[k] / num[k] for k, _ in enumerate(acc)]
        position_loss = [position_loss[k] / num[k] for k, _ in enumerate(position_loss)]
        speed_loss = [speed_loss[k] / num[k] for k, _ in enumerate(speed_loss)]

        print("classifier accuracy is ", acc)
        print("the average loss is ", position_loss)
        print("speed loss is ", speed_loss)
        with open("test_loss_new_{}.json".format(WINDOW), 'w') as f:
            json.dump({"position": position_loss, "speed": speed_loss}, f)

        if position_loss[-1] < min_loss:
            with open("route_test_loss_{}.json".format(WINDOW), 'w') as f:
                json.dump({"position": position_loss, "speed": speed_loss}, f)
        
            torch.save(self.g_net, 'generator_new_{}.pth'.format(WINDOW))
            torch.save(self.classifier, 'classifier_new_{}.pth'.format(WINDOW))
            torch.save(self.d_net, 'discriminator_{}.pth'.format(WINDOW))
            # with open('test_log.json', 'w') as f:
            #     json.dump({'epoch': epoch, 'acc': acc, 'position_loss': position_loss, 'speed_loss': speed_loss}, f)
            min_loss = position_loss[-1]
        # self.test(0, summary)

    def test(
            self,
            epoch,
            summary=None
    ):
        # self.g_net = self.g_net
        # self.g_net.eval()
        # self.classifier.eval()
        # classifier = torch.load('classifier.pth')
        # generator = torch.load('generator.pth')
        # self.g_net = generator
        # self.g_net = self.generator
        self.g_net.eval()
        # self.classifier = classifier
        self.classifier.eval()
        with torch.no_grad():
            test_loss = 0
            num = 0
            n = 0
            for data in self.test_data:
                n += 1
                if n != 5:
                    continue
                grid, vid, t_0 = data  # [1, n_car, n_grid, input_shape]
                vid = [v[0] for v in vid]
                _input = torch.zeros((WINDOW, grid.shape[1], grid.shape[2],
                                      grid.shape[3]))  # [window, n_car, n_grid, input_shape]
                _input[-1] = grid[0]
                # _input[0], _input[1], _input[2], _input[3], _input[4] = grid[0], grid[0], grid[0], grid[0], \
                #                                                         grid[0]
                _input = torch.permute(_input, (1, 0, 2, 3))
                _input = _input.cuda()

                # x, y, road_id, lane_num, position_lane, a, v_s, h_s
                real = self.vehicle_state.get_some_vehicles_state(int(t_0), vid)
                real = np.array(real)

                error_list = np.where(real[:, 2] < 0)
                real = np.delete(real, error_list, axis=0)
                _input = del_tensor(_input, error_list)

                position = np.zeros((_input.shape[0], 3))
                position_real = np.zeros((_input.shape[0], 3))

                # generator 的输入格式 [n_car, t, n_grid, input_shape]
                fake_o = self.g_net(_input)
                fake_c = self.classifier(_input)

                fake_t = torch.zeros((fake_o.shape[0], 1 + fake_o.shape[1])).cuda()
                fake_t[:, 0], fake_t[:, 1:] = fake_c.argmax(1), fake_o

                real_t = np.zeros((fake_t.shape[0], fake_t.shape[1]))
                real_t[:, :] = real[:, 3:6]  # x, y
                real_t = torch.tensor(real_t, dtype=torch.float32).cuda()

                position[:, 2] = real[:, 2]
                position[:, :2] = fake_t.detach().cpu().numpy()[:, :2]
                position_real[:, 2] = real[:, 2]
                position_real[:, :2] = real_t.detach().cpu().numpy()[:, :2]

                # [n_car, 2]
                position = np.array(self.coordinate_conversion.converse_some_conversion(position))
                position_real = np.array(self.coordinate_conversion.converse_some_conversion(position_real))
                test_loss += np.mean(
                    np.sqrt(
                        (position[:, 0] - position_real[:, 0]) ** 2 + (position[:, 1] - position_real[:, 1]) ** 2
                    )
                )
                num += 1

            # print("learning rate is {}".format(self.scheduler_g.get_last_lr()))
            print("the position error is {}".format(test_loss / num))
            print(num)
            # print("speed loss is {}".format(speed_loss / n))
            if summary is not None:
                summary.add_scalar("test", test_loss / num, epoch)


if __name__ == '__main__':
    route = ROUTE(t=WINDOW)
    # route.multi_test()
    # t1 = time.time()
    route.train_multi_second()
    # t2 = time.time()
    # print("run time is {}".format(t2 - t1))
    # route.train_model()
