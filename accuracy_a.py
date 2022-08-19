import json
import math
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_process.change_road_model import ChangeRoad
from data_process.config import grid_temporal_cache
from data_process.coordinate_conversion import CoordinateConversion
from data_process.datasets import VehiclesData
from data_process.grid_map import get_grid_from_map

from data_process.model import Generator, Discriminator
import numpy as np
import random

from data_process.road_map import RoadMap
from data_process.tools import judge_road_junction
from data_process.traffic.get_traffic_flow import GetTrafficFlow
from data_process.traffic_light import TrafficLightSet
from data_process.vehicle_state import VehicleState

T = 30
WINDOW = 5
min_loss = 100


def mse(arr_1: list, arr_2: list):
    arr_1 = np.array(arr_1)
    arr_2 = np.array(arr_2)
    mse_loss = np.sqrt(
        np.mean(
            (arr_1 - arr_2) ** 2
        )
    )
    return mse_loss


def mae(arr_1: list, arr_2: list):
    arr_1 = np.array(arr_1)
    arr_2 = np.array(arr_2)
    mae_loss = np.mean(
        np.abs(arr_1 - arr_2)
    )
    return mae_loss


def mre(arr_1: list, arr_2: list):
    arr_1 = np.array(arr_1)
    arr_2 = np.array(arr_2)
    mre_loss = np.mean(
        np.abs(arr_1 - arr_2) / arr_1
    )
    return mre_loss


def aa(arr_1: list, arr_2: list):
    arr_1 = np.array(arr_1)
    arr_2 = np.array(arr_2)
    aa = np.mean(
        1 - np.abs(
            arr_1 - arr_2
        ) / 30
    )
    return aa


# classifier accuracy is  [0.9823514088988304, 0.9809595286846161, 0.98014085739851, 0.9783040821552277, 0.9783849805593491]
# mse loss is  [0.32243051193654537, 0.32346780262887476, 0.3356475055217743, 0.3451106548309326, 0.3611458674073219]
# the average loss is  [0.30461506393410087, 0.309900677668722, 0.31718747125848445, 0.32720938941417044, 0.3358725996555286]
# speed loss is  [0.42656473740935325, 0.4311582028865814, 0.4430271804332733, 0.45274779498577117, 0.46375667527318]

# 车辆距离路口的最近距离，如果与路口的距离小于5，那么表示车辆处于三个状态
# 1. 到达目的地
# 2. 红灯亮：等待红灯（正常状态）
# 3. 绿灯亮：选择下一条道路（不参与下一秒的预测，但是需要计算车辆在路口中存在的时间，并在路口中存在时间结束后加入到预测序列中）
MIN_DISTANCE = 5

INTERSECTION_TIME = 2  # 车辆在交叉路口停留的时间，SUMO中都是1-3S
LENGTH = 500
sigma = 0.6

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
        gird_history = grid[vehicle]
        if grid_now[key] != 0:
            gird_history[0], gird_history[1], gird_history[2], gird_history[3], gird_history[4] = \
                gird_history[1], gird_history[2], gird_history[3], gird_history[4], grid_now[key]
            # gird_history[-1] = grid_now[key]
        else:
            gird_history[0], gird_history[1], gird_history[2], gird_history[3], gird_history[4] = \
                gird_history[1], gird_history[2], gird_history[3], gird_history[4], np.zeros((27, 9)).tolist()
            # gird_history[-1] = np.zeros((27, 9)).tolist()
        grid[vehicle] = gird_history
    # with open(grid_temporal_cache, 'w') as f:
    #     json.dump(grid, f)
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
    grid = np.zeros((len(vid), 5, n_grid, 9))
    # with open(grid_temporal_cache, 'r') as f:
    #     grid_history = json.load(f)
    if cache is None:
        grid_history = grid_cache
    else:
        grid_history = grid_test_cache
    for i, _ in enumerate(vid):
        vehicle = vid[i]
        grid_s = np.zeros((5, n_grid, 9))
        grid_v = np.array(grid_history[vehicle])
        for k in range(grid_v.shape[0]):
            for j in range(grid_v.shape[1]):
                grid_s[k, j] = np.array(grid_v[k, j])
        # grid_v = np.reshape(grid_v, (grid_v.shape[0], grid_v.shape[1], -1))
        grid[i] = grid_s

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


# best n_spatial_layer=3, n_temporal_layer=3, emb_dim=64, n_head=8
class ROUTE(nn.Module):
    def __init__(self, n_input=9, output_dim=2, t=5, n_grid=27, n_spatial_layer=3, n_temporal_layer=3, emb_dim=64,
                 n_head=16, drop_rate=0.1,
                 learning_rate=0.0004, epochs=500):
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

        self.milestones = [50, 100, 150, 200, 150, 300, 400, 500, 600, 700, 800, 900, 999]
        self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(self.opt_g, milestones=self.milestones, gamma=0.5)
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR(self.opt_d, milestones=self.milestones, gamma=0.5)
        self.scheduler_c = torch.optim.lr_scheduler.MultiStepLR(self.opt_c, milestones=self.milestones, gamma=0.5)

        self.criterion_minimize = nn.MSELoss().cuda()
        self.criterion = nn.BCELoss().cuda()
        self.criterion_classifier = nn.NLLLoss().cuda()

        self.vehicle_state = VehicleState()  # 获取车辆的状态真实值
        self.tls = TrafficLightSet()
        self.change_road = torch.load('change_road_model.pth')  # 换道模型
        # self.train_data = dataloader(train=True)
        self.test_data = dataloader(train=False)
        self.coordinate_conversion = CoordinateConversion()
        self.road_map = RoadMap()
        self.traffic = GetTrafficFlow("assets/vehicles.json", "assets/road_map.json")

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
            speed = speed + np.random.poisson(lam=sigma) / 5
            road_id = vehicle_states[key, 2]
            road_name = self.road_map.get_road_name(int(road_id))
            lane_num = output[key, 0]
            position_in_lane = output[key, 1]
            position_in_lane = position_in_lane + np.random.poisson(lam=sigma) / 5
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

    def evaluate_flow(self, output: np.ndarray, vid: np.ndarray, vehicle_states: np.ndarray, t: int):
        # 将数据转化为路网的格式
        # output_type : [n_car, 3]  [lane_num, lane_position, speed]
        # vid : vehicle id list
        # vehicle_states : x, y, road_id, lane_num, position_lane, a, v_s, h_s
        assert output.shape[0] == vid.shape[0] and vid.shape[0] == vehicle_states.shape[0], "vehicle number need be " \
                                                                                            "consistent "
        # [position_lane, lateral_position, s, d, angle, lane_number, light, g_s, road_id, g_d]
        vehicles = {}
        road_list = vehicle_states[:, 2]
        road_set = list(set(road_list))  # 去除重复的id
        # print(road_set)
        road_name_set = [self.road_map.get_road_name(int(rid)) for rid in road_set]  # 道路名集合

        # 初始化字典
        for rname in self.road_map.roads:
            vehicles[rname] = {
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
            vehicles[road_name]['vehicles'][vehicle] = vinfo

        # print(vehicles)
        traffic = {}
        for road_name in vehicles:  # 获取每条道路中的车辆
            if judge_road_junction(road_name):
                # 如果道路存在id则获取id，如果是交叉路口那么直接使用交叉路口名
                r = vehicles[road_name]['vehicles']
                vehicle_number = 0
                speed = 0
                for vid in r:  # 获取每个车辆的信息
                    vehicle_number += 1
                    vehicle = r[vid]
                    v_s = vehicle['info']['speed']
                    speed += float(v_s)
                mean_speed = speed / vehicle_number if vehicle_number != 0 else 0
                rid = self.road_map.get_road_id(road_name)
                traffic[rid] = {
                    "traffic_flow": vehicle_number,
                    "traffic_speed": mean_speed
                }

        traffic_flow = []
        traffic_speed = []
        # print(traffic)
        # for val in traffic:
        #     f = []
        #     s = []
        for key in traffic:
            traffic_flow.append(traffic[key]['traffic_flow'])
            traffic_speed.append(traffic[key]['traffic_speed'])
            # for key in val:
            #     f[int(key)] = val[key]['traffic_flow']
            #     s[int(key)] = val[key]['traffic_speed']
            # traffic_flow.append(f)
            # traffic_speed.append(s)

        real_f, real_s = self.traffic.get_one_data(t)
        print(t)
        print(real_f)
        print(traffic_flow)
        print(real_s)
        print(traffic_speed)
        print(math.fsum(traffic_flow))
        print(math.fsum(real_f))
        f_loss = mse(real_f, traffic_flow)
        s_loss = mse(real_s, traffic_speed)

        mae_f = mae(real_f, traffic_flow)
        mae_s = mae(real_s, traffic_speed)

        mre_f = mre(real_f, traffic_flow)
        mre_s = mre(real_s, traffic_speed)

        aa_f = aa(real_f, traffic_flow)
        aa_s = aa(real_s, traffic_speed)
        print(f_loss)
        print(s_loss)
        return f_loss, s_loss, mae_f, mae_s, mre_f, mre_s, aa_f, aa_s

    def predict(self, _input: torch.Tensor):
        pass

    def optimize_model(self, _input: torch.Tensor):
        pass

    def train_model(self):
        """
        训练模型，训练模型的过程首先将一秒一秒的训练
        如果预测下一秒的误差太大会很影响最终模型的结果
        :return:
        """
        summary = SummaryWriter('route_log')

        # 首先进行100次预训练，只训练预测一秒
        for epoch in range(2000):
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

                d_loss = (d_real_loss + d_fake_loss) * 0.5
                d_loss.backward()

                self.opt_d.step()

                position[:, 2] = real[:, 2]
                position[:, :2] = fake_t.detach().cpu().numpy()[:, :2]
                position_real[:, 2] = real[:, 2]
                position_real[:, :2] = real_t.detach().cpu().numpy()[:, :2]

                # [n_car, 2]
                position = np.array(self.coordinate_conversion.converse_some_conversion(position))
                position_real = np.array(self.coordinate_conversion.converse_some_conversion(position_real))
                position_loss += np.sqrt(
                    np.mean(
                        (position[:, 0] - position_real[:, 0]) ** 2 + (position[:, 1] - position_real[:, 1]) ** 2))
                speed = fake_o.detach().cpu().numpy()[:, -1]
                speed_real = real_t.detach().cpu().numpy()[:, -1]
                speed_loss += np.mean(
                    np.abs(speed - speed_real)
                )

                # 平均每训练三次判别器，训练一次生成器
                if random.uniform(0, 1) < 0:
                    self.opt_g.zero_grad()
                    d_fake_1 = self.d_net(_input, fake_t)
                    g_loss = self.criterion(d_fake_1, real_label)
                    g_loss.backward()
                    self.opt_g.step()
                    loss_g += g_loss
                    num_g += 1
                else:
                    self.opt_g.zero_grad()
                    g_mse_loss = self.criterion_minimize(fake_o, real_o)
                    g_mse_loss.backward()
                    self.opt_g.step()
                    mse_loss += g_mse_loss
                    num_mse += 1

                self.opt_c.zero_grad()
                nl_loss = self.criterion_classifier(fake_c, real_c.long())
                nl_loss.backward()
                self.opt_c.step()

                acc += torch.sum(fake_c.argmax(1) == real_c)
                acc_num += fake_c.shape[0]

                loss_d += d_fake_loss + d_real_loss

                car_num += 1

            self.test(epoch, summary)
            print(all)
            num += 1

            self.scheduler_d.step()
            self.scheduler_g.step()
            self.scheduler_c.step()

            print("epoch {} -------------------".format(epoch))
            print("classifier accuracy is {}".format(acc / acc_num))
            # print("learning rate is {}".format(self.scheduler_g.get_last_lr()))
            print("mse loss is {}".format(mse_loss / num_mse))
            print("the average loss is {}".format(position_loss / car_num))
            print("loss_d is {}".format(loss_d / num))
            print("loss_g is {}".format(loss_g / num_g))
            print("speed loss is {}".format(speed_loss / car_num))
            print()

            summary.add_scalar("position_loss", position_loss / car_num, epoch)
            summary.add_scalar("loss_d", loss_d / num, epoch)
            summary.add_scalar("loss_g", loss_g / num_g, epoch)
            summary.add_scalar("mse_loss", mse_loss / num_mse, epoch)
            summary.add_scalar("acc", acc / acc_num, epoch)
            summary.add_scalar("speed", speed_loss / car_num, epoch)

        torch.save(self.d_net, 'discriminator.pth')
        torch.save(self.g_net, 'generator.pth')
        torch.save(self.classifier, 'classifier.pth')
        # self.train_multi_second()
        summary.close()
        # for data in self.test_data:
        #     print(data)

    def train_multi_second(self):
        """
        只训练5s时间长度的模型，因为时间窗口长度为5s
        :return:
        """
        summary = SummaryWriter('multi_second')

        length = 6  # 训练窗口的大小

        self.classifier = torch.load('classifier.pth')
        self.generator = torch.load('generator.pth')

        self.classifier.cuda()
        self.generator.cuda()

        # 多秒的训练不用判别器
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.00002
        )

        opt_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=0.00002
        )

        milestones_1 = [20, 80, 200, 500, 800, 1200, 1600, 2000, 2500]
        milestones_2 = [30, 60, 90]

        scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=milestones_1, gamma=0.6)
        scheduler_c = torch.optim.lr_scheduler.MultiStepLR(opt_c, milestones=milestones_2, gamma=0.6)

        criterion_minimize = nn.MSELoss().cuda()
        criterion_classifier = nn.NLLLoss().cuda()

        for epoch in range(5000):
            num = [0 for _ in range(length)]
            mse_loss = [0 for _ in range(length)]
            acc = [0 for _ in range(length)]
            speed_loss = [0 for _ in range(length)]
            position_loss = [0 for _ in range(length)]

            # vid = np.array(vid)
            # vid = np.delete(vid, error_list)

            n = 0

            for data in self.train_data:
                n += 1
                # if n % length == 0:  # 保证输入和输出之间的唯一映射
                #     continue
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
                        grid = load_grid(vid, 27)
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
                    fake_o = self.generator(_input)
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

                    # 更新
                    g_mse_loss = criterion_minimize(fake_o, real_o)
                    opt_g.zero_grad()
                    g_mse_loss.backward()
                    opt_g.step()

                    # 当epoch为奇数时训练0， 偶数训练1

                    mse_loss[i] += g_mse_loss.view(-1).item()

                    # 更新
                    nl_loss = criterion_classifier(fake_c, real_c.long())
                    opt_c.zero_grad()
                    nl_loss.backward()
                    opt_c.step()

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

            scheduler_g.step()
            scheduler_c.step()

            print("epoch {} -------------------".format(epoch))
            print("classifier accuracy is ", acc)
            print("mse loss is ", mse_loss)
            print("the average loss is ", position_loss)
            print("speed loss is ", speed_loss)
            with open('train_log.json', 'w') as f:
                json.dump({'epoch': epoch, 'acc': acc, 'position_loss': position_loss, 'speed_loss': speed_loss}, f)

            if epoch > 50:
                self.multi_test()
            # self.test(epoch, summary)
            # if epoch % 30 == 0:
            #     torch.save(self.generator, 'generator.pth')
            #     torch.save(self.classifier, 'classifier.pth')

    def train_multi_second_2(self):
        """
        只训练5s时间长度的模型，因为时间窗口长度为5s
        :return:
        """
        summary = SummaryWriter('multi_second')

        length = 5  # 训练窗口的大小

        self.classifier = torch.load('classifier.pth')
        self.generator = torch.load('generator.pth')

        self.classifier.cuda()
        self.generator.cuda()

        # 多秒的训练不用判别器
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0001
        )

        opt_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=0.0001
        )

        milestones_1 = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        milestones_2 = [30, 60, 90]

        scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=milestones_1, gamma=0.6)
        scheduler_c = torch.optim.lr_scheduler.MultiStepLR(opt_c, milestones=milestones_2, gamma=0.6)

        criterion_minimize = nn.MSELoss().cuda()
        criterion_classifier = nn.NLLLoss().cuda()

        for epoch in range(200):
            num = [0 for _ in range(length)]
            mse_loss = [0 for _ in range(length)]
            acc = [0 for _ in range(length)]
            speed_loss = [0 for _ in range(length)]
            position_loss = [0 for _ in range(length)]

            for data in self.train_data:
                vid = []
                grid, vid, t_0 = data
                vid = [v[0] for v in vid]
                vid = np.array(vid)
                vehicles_in_intersection = {}  # 保存换道中的车辆信息，vid、交叉路口中停留的时间
                # x, y, road_id, lane_num, position_in_lane, v_s, a, h_s
                initial_info = self.vehicle_state.get_some_vehicles_state(int(t_0) - 1, vid)  # 获取初始车辆的信息
                road_list = {}  # 记录每个车辆当前的所在RSG编号
                direction_list = {}  # 保存当前车辆的下一次转向方向
                light_state = {}  # 保存当前红路灯状态
                grid = grid[0]
                need_del = []
                for key, _ in enumerate(initial_info):
                    vehicle = vid[key]
                    info = initial_info[key]
                    if info[2] == -1:
                        need_del.append(key)
                    road_list[vehicle] = info[2]

                if len(need_del) != 0:
                    need_del = np.array(need_del)
                    grid = del_tensor(grid, need_del)
                    vid = np.delete(vid, need_del)

                for i in range(length):
                    _input = torch.zeros((WINDOW, grid.shape[0], grid.shape[1],
                                          grid.shape[2]))  # [window, n_car, n_grid, input_shape]

                    if i == 0:
                        _input[-1] = grid
                        _input = torch.permute(_input, (1, 0, 2, 3))
                        save(np.array(_input), vid)
                    else:  # 从文件中加载grid
                        grid = load_grid(vid, 27)
                        _input = torch.tensor(grid, dtype=torch.float32)

                    _input = _input.cuda()

                    for key, _ in enumerate(vid):
                        vehicle = vid[key]
                        a = _input[key]
                        v_g = a[-1]
                        v_t = v_g[13]
                        direction_list[vehicle] = v_t[3].item()
                        lane, junction = self.road_map.next_road(int(road_list[vehicle]), int(v_t[3].item()))
                        if junction is None:
                            light_state[vehicle] = 2
                        else:
                            light = self.tls.get_current_state(int(t_0) + i - 1, lane, junction.split('_')[0])
                            light_state[vehicle] = light

                    real = self.vehicle_state.get_some_vehicles_state(int(t_0) + i, vid)
                    real = np.array(real)

                    intersection_list = np.where(real[:, 2] == -1)[0]  # 即将进入交叉路口的车辆
                    intersection_list = [vid[i] for i in intersection_list]  # 保存即将进入交叉路口的车辆的ID

                    # 将需要进入交叉路口的车辆信息保存
                    for k, _ in enumerate(intersection_list):
                        vehicle = intersection_list[k]
                        vehicles_in_intersection[vehicle] = {
                            't': 2,
                        }

                    need_del_list = np.where(real[:, 2] == -2)[0]  # 已经到达目的地的车辆
                    vid = np.delete(vid, need_del_list)
                    _input = del_tensor(_input, need_del_list)
                    real = np.delete(real, np.where(real[:, 2] == -2), axis=0)

                    # 再删除进入交叉路口的
                    vid = np.delete(vid, np.where(real[:, 2] == -1))
                    _input = del_tensor(_input, np.where(real[:, 2] == -1))
                    real = np.delete(real, np.where(real[:, 2] == -1), axis=0)

                    # generator 的输入格式 [n_car, t, n_grid, input_shape]
                    fake_o = self.generator(_input)
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
                    position_loss[i] += np.sqrt(
                        np.mean(
                            (position[:, 0] - position_real[:, 0]) ** 2 + (position[:, 1] - position_real[:, 1]) ** 2
                        )
                    )
                    num[i] += 1

                    speed = fake_o.detach().cpu().numpy()[:, -1]
                    speed_real = real_t.detach().cpu().numpy()[:, -1]
                    speed_loss[i] += np.mean(
                        np.abs(speed - speed_real)
                    )

                    # 更新
                    g_mse_loss = criterion_minimize(fake_o, real_o)
                    opt_g.zero_grad()
                    g_mse_loss.backward()
                    opt_g.step()

                    # 当epoch为奇数时训练0， 偶数训练1

                    mse_loss[i] += g_mse_loss.view(-1).item()

                    # 更新
                    nl_loss = criterion_classifier(fake_c, real_c.long())
                    opt_c.zero_grad()
                    nl_loss.backward()
                    opt_c.step()

                    acc[i] += (torch.sum(fake_c.argmax(1) == real_c) / fake_c.shape[0]).view(-1).item()

                    # 更新新的网格信息
                    vehicles_in_intersection, out_inter = update_intersection_vehicle(vehicles_in_intersection)
                    # x, y, road_id, lane_num, position_lane, a, v_s, h_s
                    output = fake_t.detach().cpu().numpy()  # 使用上一秒预测的数据
                    real_fake = np.zeros((real.shape[0], real.shape[1]))
                    # 将当前出交叉路口的车辆放入id列表中
                    for vehicle in out_inter:
                        # 当前需要准备的数据是t_0 + i时间的数据
                        # 50s 的数据 51s
                        v_s = self.vehicle_state.get_one_vehicle_state(int(t_0) + i, vehicle)
                        # l_t = np.zeros((1, len(v_s)))
                        # 如果当前车辆还在交叉路口中，那么继续将车辆放入到字典中
                        if type(v_s[2]) == str:
                            vehicles_in_intersection[vehicle] = {
                                't': 1,
                            }
                            continue
                        l = np.array(v_s)
                        l = np.reshape(l, (1, -1))
                        l_t = l[:, 3:6]
                        road_list[vehicle] = l[0, 2]

                        vid = np.append(vid, vehicle)
                        real_fake = np.append(real_fake, l, axis=0)
                        output = np.append(output, l_t, axis=0)

                    # 将real_fake内容补齐，real_fake只用到它的道路id
                    for k, _ in enumerate(vid):
                        vehicle = vid[k]
                        real_fake[k, 2] = road_list[vehicle]

                    grip_now, vehicles_id = self.make_grid(output, vid, real_fake, int(t_0) + i)
                    update_grid(grip_now, vehicles_id)

            # self.multi_test(epoch, summary)
            acc = [acc[k] / num[k] for k, _ in enumerate(acc)]
            position_loss = [position_loss[k] / num[k] for k, _ in enumerate(position_loss)]
            speed_loss = [speed_loss[k] / num[k] for k, _ in enumerate(speed_loss)]
            mse_loss = [mse_loss[k] / num[k] for k, _ in enumerate(mse_loss)]

            scheduler_g.step()
            scheduler_c.step()

            print("epoch {} -------------------".format(epoch))
            print("classifier accuracy is ", acc)
            print("mse loss is ", mse_loss)
            print("the average loss is ", position_loss)
            print("speed loss is ", speed_loss)

            # self.multi_test()
            # self.test(epoch, summary)

        torch.save(self.generator, 'generator.pth')
        torch.save(self.classifier, 'classifier.pth')

    def multi_test(self, epoch=0):
        """
        多秒的测试模型，也可以作为最后的预测部分
        :return:
        """
        length = 30  # 测试时间长度的大小
        global min_loss
        summary = SummaryWriter('multi_second')
        classifier = torch.load('classifier.pth')
        generator = torch.load('generator.pth')
        # classifier = self.classifier
        # generator = self.generator

        num = [0 for _ in range(length)]
        acc = [0 for _ in range(length)]
        speed_loss = [0 for _ in range(length)]
        position_loss = [0 for _ in range(length)]
        flow_mse = []
        speed_mse = []

        flow_mae = []
        speed_mae = []

        flow_mre = []
        speed_mre = []

        flow_aa = []
        speed_aa = []
        n = 0

        for data in self.test_data:
            # s_y, s, d, angle, s_lane, light, g_s, road_id, g_d
            n += 1
            if n != 10:
                continue
            grid, vid, t_0 = data
            vid = [v[0] for v in vid]
            vid = np.array(vid)
            vehicles_in_intersection = {}  # 保存换道中的车辆信息，vid、交叉路口中停留的时间
            # x, y, road_id, lane_num, position_in_lane, v_s, a, h_s
            initial_info = self.vehicle_state.get_some_vehicles_state(int(t_0) - 1, vid)  # 获取初始车辆的信息
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
                        grid = load_grid(vid, 27, cache='test')
                        _input = torch.tensor(grid, dtype=torch.float32)

                    _input = _input.cuda()

                    for key, _ in enumerate(vid):
                        vehicle = vid[key]
                        # a = _input[key]
                        # v_g = a[68888888888]
                        # v_t = v_g[13]
                        target_road = self.vehicle_state.get_vehicle_trip(vehicle)[-1]
                        target_road = self.road_map.get_road_id(target_road)
                        current_road = road_list[vehicle]
                        aim_info = torch.tensor(np.array([[current_road, target_road]]),
                                                dtype=torch.float32)
                        speed_list[vehicle] = now_state[key, -1]
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
                            print(lane, angle, junction, current_road, road_list[vehicle])
                            print(vehicle)

                            light_state[vehicle] = self.tls.get_current_state(t, junction.split('_')[0],
                                                                              junction) if target_road != current_road else 2
                            direction_list[vehicle] = angle

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

                    acc[i] += (torch.sum(fake_c.argmax(1) == real_c) / fake_c.shape[0]).view(-1).item()

                    # 预测完之后，需要更新数据
                    # 将交叉路口中的车辆停留时间数减一
                    # 更新新的网格信息
                    vehicles_in_intersection, out_inter = update_intersection_vehicle(vehicles_in_intersection,
                                                                                      cache='test')
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
                    f_loss, s_loss, mae_f, mae_s, mre_f, mre_s, aa_f, aa_s = self.evaluate_flow(output, vid, real_fake,
                                                                                                int(t_0) + i)
                    flow_mse.append(f_loss)
                    speed_mse.append(s_loss)
                    flow_mae.append(mae_f)
                    speed_mae.append(mae_s)
                    flow_mre.append(mre_f)
                    speed_mre.append(mre_s)
                    flow_aa.append(aa_f)
                    speed_aa.append(aa_s)
                    update_grid(grip_now, vehicles_id, cache="test")

                    vid = np.array(vehicles_id)
                    a = np.array(grip_now)
                    a_t = a[:, 13]
                    now_state = np.zeros((vid.shape[0], 3))
                    now_state[:, 1] = a_t[:, 0]
                    now_state[:, 0] = a_t[:, 4]
                    now_state[:, 2] = a_t[:, 1]

        acc = [acc[k] / num[k] for k, _ in enumerate(acc)]
        position_loss = [position_loss[k] / num[k] for k, _ in enumerate(position_loss)]
        speed_loss = [speed_loss[k] / num[k] for k, _ in enumerate(speed_loss)]

        # print("classifier accuracy is ", acc)
        # print("the average loss is ", position_loss)
        # print("speed loss is ", speed_loss)

        print("flow mse is {}".format(flow_mse))
        print("speed mse is {}".format(speed_mse))
        print("flow mae is {}".format(flow_mae))
        print("speed mae is {}".format(speed_mae))
        print("flow mre is {}".format(flow_mre))
        print("speed mre is {}".format(speed_mre))
        print("flow aa is {}".format(flow_aa))
        print("speed aa is {}".format(speed_aa))
        if os.path.isfile('route_traffic_p.json'):
            with open('route_traffic_p.json', 'r') as f:
                poisson = json.load(f)
                poisson[sigma] = {
                    'flow_mse': flow_mse[-1], 'speed_mse': speed_mse[-1],
                    "flow_mae": flow_mae[-1], 'speed_mae': speed_mae[-1],
                    "flow_aa": flow_aa[-1], "speed_aa": speed_aa[-1]
                }
            with open('route_traffic_p.json', 'w') as f:
                json.dump(poisson, f)
        else:
            with open('route_traffic_p.json', 'w') as f:
                json.dump({sigma: {
                    'flow_mse': flow_mse[-1], 'speed_mse': speed_mse[-1],
                    "flow_mae": flow_mae[-1], 'speed_mae': speed_mae[-1],
                    "flow_aa": flow_aa[-1], "speed_aa": speed_aa[-1]
                }
                }, f)

        # if position_loss[-1] < min_loss:
        #     torch.save(generator, 'models/generator_test.pth')
        #     torch.save(classifier, 'models/classifier_test.pth')
        #     min_loss = position_loss[-1]
        self.test(0, summary)

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
        self.g_net.eval()
        # self.classifier = classifier
        self.classifier.eval()
        with torch.no_grad():
            test_loss = 0
            num = 0
            n = 0
            for data in self.test_data:
                n += 1
                if n != 10:
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
            print("the position error is {}".format(test_loss / n))
            print(num)
            # print("speed loss is {}".format(speed_loss / n))
            if summary is not None:
                summary.add_scalar("test", test_loss / num, epoch)


if __name__ == '__main__':
    route = ROUTE()
    # route.train_multi_second_2()
    route.multi_test()
    # route.train_multi_second()
    # route.train_model()
    # route.test(0)
# frag_num = 1
# frag_size = grid.shape[1] // frag_num
# for i in range(1, frag_num + 1):
#     _grid = grid[0][(i - 1) * frag_size: i * frag_size] if i != frag_num else grid[0][
#                                                                               (i - 1) * frag_size:]
#     if _grid.shape[0] == 0:
#         continue
#     _vid = vid[(i - 1) * frag_size: i * frag_size] if i != frag_num else vid[(i - 1) * frag_size:]
#     _input = torch.zeros((WINDOW, _grid.shape[0], _grid.shape[1],
#                           _grid.shape[2]))  # [window, n_car, n_grid, input_shape]

# position = np.zeros((_grid.shape[0], 3))
# position_real = np.zeros((_grid.shape[0], 3))
# _input[0], _input[1], _input[2], _input[3], _input[4] = \
#     _grid, _grid, _grid, _grid, _grid

# classifier accuracy is  [0.9821343362331391, 0.9612777769565582, 0.9386183798313141, 0.92265585064888, 0.9080397963523865]
# the average loss is  [0.7785852779570455, 2.905305014797135, 3.857641728571835, 4.570430045482774, 6.288938476696958]
# speed loss is  [0.5514894902706147, 0.9274308204650878, 1.2169042825698853, 1.3975398063659668, 1.5751713752746581]

# [0.9359406826456553, 1.0448508863345756, 1.4799216296765243, 1.9289972649145266, 1.6677968921132995, 2.0832759071790923, 2.413308668914486, 2.353936256042348, 2.3217629957377737, 2.6112167034129463, 2.6937323259398402, 2.7706923106499453, 3.1475814575897005, 3.186680296054661, 3.715029219761653, 4.394943454729593, 5.210718358269021, 6.034205752258114, 6.415240569058434, 5.811442546164118, 5.526884403060554, 5.43183487700935, 5.459082892114118, 5.47646519903616, 5.643219081979315, 5.945223825032922, 5.8919440704905774, 5.713430390085891, 5.876971869931745, 6.965280307115152]
# speed loss is  [0.3240492641925812, 0.3784528970718384, 0.4741676449775696, 0.4532112777233124, 0.38466885685920715, 0.3169536292552948, 0.2993893027305603, 0.2853514552116394, 0.30058032274246216, 0.2969183623790741, 0.3496602177619934, 0.37903863191604614, 0.3604821562767029, 0.3847208321094513, 0.4688124358654022, 0.4719659686088562, 0.5222856402397156, 0.5215932130813599, 0.5274667739868164, 0.5452609062194824, 0.6055927872657776, 0.5224320292472839, 0.5921021699905396, 0.6017799377441406, 0.6018346548080444, 0.5695414543151855, 0.5950250029563904, 0.6237910389900208, 0.832759439945221, 1.2004899978637695]
