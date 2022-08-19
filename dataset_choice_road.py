import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from data_process.config import road_map_file, vehicles_file, traffic_light_file
from data_process.road_map import RoadMap
from data_process.tools import judge_road_junction, get_road_name
import numpy as np


class ChoiceRoad:
    """
    处理模型选择的问题
    """

    def __init__(
            self,
            vehicle_file,
            map_file,
    ):
        self.vehicle_file = vehicle_file
        self.map_file = map_file
        self.vehicles = self.load_files()
        self.map = RoadMap(self.map_file)
        self.change_road_info, self.choice_road = self.generate_data()

    def load_files(self):
        # 加载文件
        with open(self.vehicle_file, 'r') as f:
            vehicle = json.load(f)

        return vehicle

    def generate_data(self):
        """
        生成数据
        得到转向-目标-道路名等信息
        :return:
        """
        # 保存一个以及遍历过的状态
        has_search = []
        info_list = []
        result_list = []
        for r_time in self.vehicles:
            info = []
            result = []
            for road_name in r_time['vehicles']:
                if not judge_road_junction(road_name):
                    continue
                # print(r_time['vehicles'])
                # print(road_name)
                vehicles = r_time['vehicles'][road_name]['vehicles']
                for vid in vehicles:
                    # if '{}-{}'.format(road_name, vid) in has_search:
                    #     continue
                    # else:
                    #     has_search.append('{}-{}'.format(road_name, vid))
                    vehicle = vehicles[vid]
                    # print(vehicle)
                    route = vehicle['info']['route']
                    target = route[-1]
                    if road_name == target:
                        next_road = road_name
                    else:
                        for i in range(len(route) - 1):
                            if route[i] == road_name:
                                next_road = route[i + 1]
                    # next_direction = self.map.next_direction(road_name, next_road)
                    # if next_direction == 3:
                    #     continue
                    if self.map.get_road_id(road_name) == self.map.get_road_id(next_road):
                        continue
                    info.append([
                        self.map.get_road_id(road_name),
                        self.map.get_road_id(target),
                    ])
                    result.append(self.map.get_road_id(next_road))
            info_list.append(info)
            result_list.append(result)
        return info_list, result_list


class MatrixRoad(Dataset):
    def __init__(self, train=True):
        self.traffic = GetTrafficFlow("assets/vehicles.json", "assets/road_map.json", load=True)
        self.light = Light()
        self.choice_road = ChoiceRoad(vehicles_file, road_map_file)
        self.traffic_flow = torch.tensor(self.traffic.get_traffic_flow(), dtype=torch.float32)
        self.traffic_speed = torch.tensor(self.traffic.get_traffic_speed(), dtype=torch.float32)
        self.traffic_light = torch.tensor(self.light.lights, dtype=torch.float32)
        self.info_list = self.choice_road.change_road_info
        self.road = self.choice_road.choice_road
        self.road_maps = self.make_data()
        self.train = train

    def get_info(self, t, j):
        info = torch.zeros((len(j), 3, self.traffic_flow.shape[1], self.traffic_flow.shape[2]),
                           dtype=torch.float32)
        l = torch.zeros((len(j), self.traffic_light.shape[1], self.traffic_light.shape[2]))
        for i, v in enumerate(j):
            # print(v)
            r_map = road_map.map.copy()
            r_map_index = road_map.map_index
            k = dict(zip(r_map_index.values(), r_map_index.keys()))
            for val in r_map_index:
                r_map[val[0], val[1]] = 5
            s = v[0]
            e = v[1]
            r_map[k[s][0], k[s][1]] = -5
            r_map[k[e][0], k[e][1]] = -10
            info[i, 0], info[i, 1], info[i, 2] = self.traffic_flow[t[i]], self.traffic_speed[t[i]], \
                                                 torch.tensor(r_map, dtype=torch.float32)
            l[i] = self.traffic_light[t[i]]

        return info, l

    def make_data(self):
        result = []
        for t in self.info_list:
            res_t = []
            for v in t:
                r_map = road_map.map.copy()
                r_map_index = road_map.map_index
                k = dict(zip(r_map_index.values(), r_map_index.keys()))
                for val in r_map_index:
                    r_map[val[0], val[1]] = 5
                s = v[0]
                e = v[1]
                m = r_map
                m[k[s][0], k[s][1]] = -5
                m[k[e][0], k[e][1]] = -10

                res_t.append(m)
            result.append(res_t)
        # print(len(result[0]))
        # print(len(self.road[0]))
        return result

    def clip_data(self):
        train_size = int(0.8 * self.traffic_flow.shape[0])
        return train_size

    def __len__(self):
        return int(len(self.traffic_flow) * 0.8)

    def __getitem__(self, index):
        # print(self.traffic_flow.shape)
        info = torch.zeros((len(self.road_maps[index]), 3, self.traffic_flow.shape[1], self.traffic_flow.shape[2]),
                           dtype=torch.float32)
        l = torch.zeros((len(self.road_maps[index]), self.traffic_light.shape[1], self.traffic_light.shape[2]))
        for i in range(len(self.road_maps[index])):
            info[i, 0], info[i, 1], info[i, 2] = self.traffic_flow[index], self.traffic_speed[index], \
                                                 torch.tensor(self.road_maps[index][i], dtype=torch.float32)
            l[i] = self.traffic_light[index]
        r = torch.tensor(self.road[index])
        info_list = torch.tensor(self.info_list[index])[:, 0]

        return info, l, r, info_list


class GetTrafficFlow:
    def __init__(
            self,
            vehicle_file,
            map_file,
            load=True
    ):
        base_path = os.getcwd()
        self.load = load
        self.vehicle_file = os.path.join(base_path, vehicle_file)
        self.map_file = os.path.join(base_path, map_file)
        self.__traffic_data = None
        self.__traffic_flow = None
        self.__traffic_speed = None
        self.__traffic_flow_map = None
        self.__traffic_speed_map = None
        if not self.load:
            self.map, self.vehicles = self.__initialize_data()
            # self.make_map_data()
        self.__get_current_data()

    def __initialize_data(self):
        with open(self.vehicle_file, 'r') as f:
            vehicles = json.load(f)

        with open(self.map_file, 'r') as f:
            maps = json.load(f)

        return maps, vehicles

    def __get_current_data(self):
        with open(traffic_map_file, 'r') as f:
            res = json.load(f)
            self.__traffic_flow_map = res['flow']
            self.__traffic_speed_map = res['speed']

    def get_traffic_flow(self):
        return self.__traffic_flow_map

    def get_traffic_speed(self):
        return self.__traffic_speed_map

    def make_map_data(self):
        flow_map = []
        speed_map = []
        # 构造交叉路口 * 交叉路口的数据
        n = 0
        for key, val in enumerate(self.__traffic_data):
            # print(key
            n += 1
            flow_maps = road_map.map.copy()
            speed_maps = np.zeros((road_map.map.shape[0], road_map.map.shape[1]), dtype=np.float32)
            # print(maps)
            for rid in val:
                traffic_flow = self.__traffic_data[key][rid]['traffic_flow']
                traffic_speed = self.__traffic_data[key][rid]['traffic_speed']
                # print(traffic_speed)
                index = road_map.get_index_by_id(int(rid))
                flow_maps[index[0]][index[1]] = traffic_flow / 10
                speed_maps[index[0]][index[1]] = traffic_speed / 10

            flow_map.append(flow_maps)
            speed_map.append(speed_maps)

        with open(traffic_map_file, 'w') as f:
            json.dump({'speed': speed_map, 'flow': flow_map}, f)

    def get_one_map(self, time):
        return self.__traffic_flow_map[time], self.__traffic_speed_map[time]

    def get_some_map(self, time):
        traffic_flow = []
        traffic_speed = []
        for t in time:
            t_f, t_s = self.get_one_map(t)
            traffic_flow.append(t_f)
            traffic_speed.append(t_s)
        return traffic_flow, traffic_speed


class Light:
    def __init__(self):
        self.tls_file = traffic_light_file
        self.original_data = self.get_original_data()
        self.lights = []
        self.load_light_map()

    def load_light_map(self):
        with open("assets/light_map.json", 'r') as f:
            self.lights = json.load(f)

    def get_original_data(self):
        filename = os.path.join(os.getcwd(), self.tls_file)
        with open(filename, 'r') as f:
            tls_data = json.load(f)
        return tls_data

    def make_light_map(self):
        length = len(self.original_data)
        for t in range(length):
            light_map = road_map.light_map.copy()
            if t % 100 == 0:
                print("当前已经加载数据：{}".format(t))
            for i in range(road_map.v_num):
                for j in range(road_map.v_num):
                    if i == j:
                        continue
                    light_map[i, j] = self.get_current_state_by_road_id(t, i, j)
            self.lights.append(light_map.tolist())

        with open("assets/light_map.json", 'w') as f:
            json.dump(self.lights, f)

    def get_current_state_by_road_id(self, time, road_id, next_road_id):
        jun_id = road_map.get_junction_between_two_road(road_id, next_road_id)
        jun_name = road_map.get_junction_name_by_id(jun_id)[1:]
        road_name = road_map.get_road_name(road_id)
        next_road_name = road_map.get_road_name(next_road_id)
        light_state = self.original_data[time][jun_name]
        tls_state = light_state['tls_state']
        controlled_links = light_state['controlled_links']
        light = 0
        for key, tls in enumerate(controlled_links):
            if len(tls) == 0:
                continue
            start = get_road_name(tls[0][0])
            end = get_road_name(tls[0][1])
            if road_name == start and next_road_name == end:
                light = 1 if tls_state[key] == 'G' or tls_state[key] == 'g' else 0
        return light


if __name__ == '__main__':
    # vehicle_state = GetTrafficFlow("assets/vehicles.json", "assets/road_map.json", load=True)
    # flow, speed = vehicle_state.get_one_map(10)
    # print(flow)
    # light = Light()
    road_map = RoadMap("assets/road_map.json")
    traffic_map_file = "assets/traffic_map.json"
    road = MatrixRoad()
    road.make_data()
