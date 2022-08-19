"""
该部分用于实现数据加载，将数据转换为Tensor张量的格式
同时分割开训练数据和测试数据
"""
import json
import os
import numpy as np
import torchvision
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from data_process.change_road_model import ChangeRoad
from data_process.config import train_road, vehicles_file, road_map_file, traffic_light_file, roads_traffic_file
from data_process.grid_map import InitMap2Grid
from data_process.road_map import RoadMap
from data_process.tools import find_in_dict, judge_road_junction


class VehicleData(Dataset):
    """
    该类用于处理车辆数据，将车辆数据json数据变成可以直接用于机器学习运算的数据
    """

    def __init__(
            self,
            vehicles_file,  # 存储车辆数据的文件名
            road_file,  # 存储路网的文件
            predict_length=5  # 预测数据的长度
    ):
        self.predict_length = predict_length
        self.vehicles_file = vehicles_file
        self.original_vehicles_data = self.get_original_data()
        self.objective_vehicles = self.get_objective_vehicle()
        self.secondary_data = self.secondary_operate_data()
        # self.map = RoadMap(road_file)
        print(json.dumps(self.secondary_data[-1]))
        # print(json.dumps(self.objective_vehicles['404.2']))

    def get_original_data(self):
        # 读取车辆数据文件
        file_name = os.path.join(os.getcwd(), self.vehicles_file)
        with open(file_name, 'r') as f:
            original_data = json.load(f)
        return original_data

    def get_objective_vehicle(self):
        """
        处理从sumo获取的原始json数据
        流程：
        1. 判断现有的车辆列表是否为空，如果为空则直接将当前车辆加入
        2. 不为空则判断是否存在当前车辆数据
                存在则将当前的数据加入当该数据中
            如果不存在
                直接将当前车辆加入
        :return: objective_vehicles
        """
        objective_vehicles = {}
        for vehicle in self.original_vehicles_data:
            vid = vehicle['id']
            time = int(vehicle['time'])
            if len(objective_vehicles) == 0:  # 目标车辆列表为空，可以将当前数据直接加入
                objective_vehicles[vid] = {}
                objective_vehicles[vid][time] = {
                    'id': vehicle['id'],
                    'in_road': vehicle['in_road'],
                    'info': vehicle['info'],
                    'neighbor': vehicle['neighbors'],
                    'time': time
                }
                continue
            if find_in_dict(objective_vehicles, vid):  # 如果列表中已经存在该目标车辆，则加入到该子集中
                if not find_in_dict(objective_vehicles[vid], time):
                    objective_vehicles[vid][time] = {
                        'id': vehicle['id'],
                        'in_road': vehicle['in_road'],
                        'info': vehicle['info'],
                        'neighbor': vehicle['neighbors'],
                        'time': time
                    }
            else:  # 如果不存在则需要创建一个数据子集
                objective_vehicles[vid] = {}
                objective_vehicles[vid][time] = {
                    'id': vehicle['id'],
                    'in_road': vehicle['in_road'],
                    'info': vehicle['info'],
                    'neighbor': vehicle['neighbors'],
                    'time': time
                }
        return objective_vehicles

    def secondary_operate_data(self):
        """
        第一次只是简单的提取原始数据，第二次需要对数据进行深入的处理
        由于第一次处理的数据中存在大量的数据是不可以直接使用的，比如目标车辆30s数据中周围车辆很有可能随着时间而发生变化
        此外车辆在数据中存在的时间必然超过了30s，因此需要对这写数据进行分割，比如将数据按s进行提取为一个独立的处理单位
        :return:
        """
        vehicles_secondary_data = []
        for key in self.objective_vehicles:
            vehicle = self.objective_vehicles[key]
            count = 0
            length = len(vehicle)
            for time in vehicle:
                sample_data = []
                current_vehicle = vehicle[time]
                if length <= count + self.predict_length + 1:  # 如果该车生成的数据时间跨度太短，则不需要该车数据
                    break
                # neighbors = vehicle['data']['neighbors']
                neighbors = [v['neighbor_id'] for v in current_vehicle['neighbor']]
                # print(vehicle)

                # 生成采样数据,需要查找每个样本点的周围车辆的数据
                for i in range(time, time + self.predict_length + 1):
                    if not find_in_dict(vehicle, i):
                        break
                    c_vehicle = {  # 获取目标车辆当前的信息
                        'id': vehicle[i]['id'],
                        'info': vehicle[i]['info'],
                        'in_road': vehicle[i]['in_road'],
                        'time': i
                    }
                    s_vehicles = list(map(lambda neighbor: {
                        'id': neighbor,
                        'info': self.objective_vehicles[neighbor][i]['info'],
                        'in_road': self.objective_vehicles[neighbor][i]['in_road'],
                        'time': i
                    } if find_in_dict(self.objective_vehicles[neighbor], i) else {
                        'id': neighbor,
                        'info': None,
                        'in_road': False,
                        'time': i
                    }, neighbors))
                    sample_data.append({
                        'object_vehicle': c_vehicle,
                        'surrounding_vehicle': s_vehicles
                    })
                vehicles_secondary_data.append(sample_data)  # 保存每个采样点
                count += 1
        print("一共有{}个样本".format(len(vehicles_secondary_data)))
        return vehicles_secondary_data

    def matriculated(self):
        """
        将二次处理的数据矩阵化，也就是最后一步操作
        矩阵化之后的数据能够直接用于机器学习
        :return:
        """
        print(json.dumps(self.secondary_data[-1]))
        for sample in self.secondary_data:
            # 对每个样本进行矩阵化
            # x, y, l, v, s, r, a, l_p, l_v, nv_l
            matrix = []
            for sample_t in sample:
                obj_road = sample_t['object_vehicle']['info']['road']
                if obj_road == train_road:
                    route = sample_t['object_vehicle']['info']['route']
                    next_road = route[route.index(obj_road) + 1] if obj_road != route[-1] else obj_road
                    obj_n = self.map.next_direction(obj_road, next_road)
                    obj_h = sample_t['object_vehicle']['info']['lateral_speed']
                else:
                    obj_n = 0
                    obj_h = 0
                obj_x = sample_t['object_vehicle']['info']['position'][0]
                obj_y = sample_t['object_vehicle']['info']['position'][1]
                obj_l = sample_t['object_vehicle']['info']['lane'].split('_')[-1]
                obj_v = sample_t['object_vehicle']['info']['speed']
                obj_a = sample_t['object_vehicle']['info']['accelerate']
                obj_is_road = sample_t['object_vehicle']['in_road']
                obj_id = sample_t['object_vehicle']['id']
                times = sample_t['object_vehicle']['time']
                matrix.append([obj_x, obj_y, obj_l, obj_v, obj_a, obj_h, obj_n, obj_is_road, obj_road, obj_id, times])
                for surrounding in sample_t['surrounding_vehicle']:
                    if surrounding['info'] is None:  # 说明该车到达终点，已经销毁
                        matrix.append([-1, -1, -1, -1, -1, -1, -1, False, '-1', surrounding['id'], surrounding['time']])
                        continue
                    road = surrounding['info']['road']
                    if road == train_road:  # 如果车辆不在道路中，则不记录下一跳方向和h
                        route = surrounding['info']['route']
                        next_road = route[route.index(road) + 1] if road != route[-1] else road
                        n = self.map.next_direction(obj_road, next_road)
                        h = surrounding['info']['lateral_speed']
                    else:
                        n = 0
                        h = 0
                    x = surrounding['info']['position'][0]
                    y = surrounding['info']['position'][1]
                    l = surrounding['info']['lane'].split('_')[-1]
                    v = surrounding['info']['speed']
                    a = surrounding['info']['accelerate']
                    is_road = surrounding['in_road']
                    id = surrounding['id']
                    matrix.append([x, y, l, v, a, h, n, is_road, road, id, times])
                print(matrix)
        pass

    def __len__(self):
        return len(self.secondary_data)

    def __getitem__(self, index):
        pass


class RoadTrafficFlow:
    pass


class VehiclesData(Dataset):
    """
    处理车辆数据，返回指定时间的路网车辆数据
    """
    __constant__ = ['__grid_data', '__vehicles_id']

    def __init__(self, train=True):
        map2grip = InitMap2Grid(vehicles_file, road_map_file, traffic_light_file, roads_traffic_file)
        self.__grid_data, self.__vehicles_id, self.start = map2grip.get_grip()
        self.__train_grid, self.__train_vehicles = \
            self.__grid_data[:int(len(self.__grid_data) * 0.8)], self.__vehicles_id[:int(len(self.__grid_data) * 0.8)]
        self.__test_grid, self.__test_vehicles = \
            self.__grid_data[int(len(self.__grid_data) * 0.8):], self.__vehicles_id[int(len(self.__grid_data) * 0.8):]
        self.train = train

    def __getitem__(self, index):
        # s_position_lane, s_y, s_h, d, angle, s_lane, light, g_s, road_id, g_d
        # s_position_lane, s_h, d, angle, s_lane, light, g_s, road_id, g_d
        if self.train:
            train_grid = torch.tensor(self.__train_grid[index], dtype=torch.float32)
            grid = torch.zeros((train_grid.shape[2] - 1, train_grid.shape[0],
                                train_grid.shape[1]))
            t = torch.permute(train_grid, (2, 0, 1))
            grid[0], grid[1:] = t[0], t[2:]  # 当前时间的路网车辆网格信息
            vehicle_id = self.__train_vehicles[index]  # 当前时间路网车辆id
        else:
            test_grid = torch.tensor(self.__test_grid[index], dtype=torch.float32)
            grid = torch.zeros((test_grid.shape[2] - 1, test_grid.shape[0],
                                test_grid.shape[1]))
            t = torch.permute(test_grid, (2, 0, 1))
            grid[0], grid[1:] = t[0], t[2:]
            vehicle_id = self.__test_vehicles[index]

        return torch.permute(grid, (1, 2, 0)), vehicle_id, self.start + self.__vehicles_id.index(vehicle_id) + 1

    def __len__(self):
        return int(len(self.__grid_data) * 0.8) if self.train else int(len(self.__grid_data) * 0.2)


def collate(batch):  # 自定义校对函数
    return batch


if __name__ == '__main__':
    # vehicles = VehicleData('vehicles.json', 'road_map.json')
    # vehicles.secondary_operate_data()
    # vehicles.matriculated()
    # road_map = RoadMap('road_map.json')
    # nex_road = road_map.next_direction('-gneE42', 'gneE46')
    # print(nex_road)
    # choice_road = ChoiceRoad('vehicles.json', 'road_map.json', train=True)
    # res = choice_road.generate_data()
    # print(res.shape)
    # loader = DataLoader(choice_road, batch_size=64, shuffle=True, drop_last=True)
    # for data in loader:
    #     print(data)
    # v_data = VehiclesData()
    grip_data = VehiclesData()
    # loader = DataLoader(grip_data, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate)
    print(grip_data[0][0].shape)
