"""
该程序用于车辆周围信息的网格化
将车辆周围的信息变成一个张量
"""
import json
import math
import os
import numpy as np
import torch

from data_process.change_road_model import ChangeRoad, ChangeRoadModel, CNN
from data_process.config import vehicles_file, road_map_file, traffic_light_file, roads_traffic_file
from data_process.road_map import RoadMap
from data_process.tools import judge_road_junction, get_lane_number
from data_process.traffic_light import TrafficLightSet

XDIM = 3
YDIM = 9
SDIM = 10

change_road = torch.load('change_road_model.pth')
change_road.cuda()
tls = TrafficLightSet()
road_map = RoadMap()


def get_grid_from_map(vehicles):
    """
    获取当前时间路网下的所有网格化数据，按照车辆ID进行保存
    :return:
    """
    # print(json.dumps(vehicles))
    vehicles_id = []
    vehicles = vehicles['vehicles']
    grip_now = []
    # print(json.dumps(vehicles))
    for road_name in vehicles:
        # 如果车辆在RSG中而不是在路口中，则可以直接生成当前车辆的信息
        # 如果车辆是在路口中，则不加入，但是需要保留
        vehicles_in_road = vehicles[road_name]['vehicles']  # 获取在每个道路中的车辆情况

        if judge_road_junction(road_name):
            for vid in vehicles_in_road:
                # 去除一些会对数据造成不利影响的数据
                # 1. 到达道路尽头且到达终点的车辆
                vehicle = vehicles_in_road[vid]
                route = vehicle['info']['route']
                target = route[-1]
                position_lane = vehicle['info']['position_in_lane']
                road_id = road_map.get_road_id(road_name)
                # if road_map.get_road_length(road_id) - position_lane < 5:

                # 遍历当前道路中的所有车辆
                vehicles_id.append(vid)
                grip = np.zeros((XDIM * YDIM, SDIM))

                position = vehicle['info']['position']
                lane_number = get_lane_number(vehicle['info']['lane'])
                lateral_position = vehicle['info']['lateral_position']
                # v_length = vehicle['info']['length']
                v_length = 7.5

                # 定义网格的边界
                grip_x = [lane_number + 1, lane_number, lane_number - 1]
                grip_y = [position_lane + (i - YDIM // 2) * v_length for i in range(YDIM)]

                s_num = 0
                for sid in vehicles_in_road:
                    if sid != vid:  # 去环境中找是否存在在网格中的车辆
                        s_vehicle = vehicles_in_road[sid]
                        s_lane = get_lane_number(s_vehicle['info']['lane'])
                        s_position_lane = s_vehicle['info']['position_in_lane']
                        s_position = s_vehicle['info']['position']
                        s_y = s_vehicle['info']['lateral_position']
                        if (grip_x[-1] <= s_lane <= grip_x[0]) \
                                and (grip_y[0] <= s_position_lane <= grip_y[-1]):
                            s_num += 1
                            i = int(s_lane - grip_x[-1])
                            j = int((s_position_lane - grip_y[0]) / v_length)
                            s_h = s_vehicle['info']['speed']  # 水平速度
                            d = math.sqrt(
                                (s_position[-1] - position[-1]) ** 2 + (s_position[0] - position[0]) ** 2)  # 与主车的距离
                            angle = 3  # 下一个转向，除了本车之外转向都置为3
                            light = 2  # 交通信号灯，0为红灯，1为绿灯，2为其他车辆占位符
                            g_s = 1  # 网格的占位状态
                            g_d = road_map.get_road_length(road_id) - s_position_lane  # 网格与路口的距离
                            # 修改数据不用s_v, a
                            grip[j * XDIM + i] = [
                                s_position_lane, s_y, s_h, d, angle, s_lane, light, g_s, road_id, g_d
                            ]

                s_h = vehicle['info']['speed']
                d = 0

                light = vehicle['info']['light']
                angle = vehicle['info']['angle']

                g_s = 1
                g_d = road_map.get_road_length(road_id) - position_lane
                grip[XDIM * YDIM // 2] = [position_lane, lateral_position, s_h, d, angle, lane_number, light,
                                          g_s, road_id, g_d]
                # print([position_lane, lateral_position, s, d, angle, lane_number, light, g_s, road_id, g_d])
                road_length = road_map.get_road_length(road_id)
                lane_num = road_map.get_lane_num(road_id)

                # 将未有车辆占据的节点进行处理
                for x in grip_x:
                    for y in grip_y:
                        i = int(x - grip_x[-1])
                        j = int((y - grip_y[0]) / v_length)
                        d = math.sqrt(
                            ((x - lane_number) * 3.2) ** 2 + (y - position_lane) ** 2
                        )  # 需要计算网格到目标车辆的距离
                        if np.sum(grip[j * XDIM + i]) == 0:
                            grip[j * XDIM + i] = [0, 0, 0, d, 0, x, 0, 0, road_id,
                                                  road_length - (position_lane + (j - YDIM // 2) * v_length)]
                        # 如果该网格不在道路内，则需要对其进行相应的处理
                        if x < 0 or x > (lane_num - 1) or y < 0 or y > road_length:
                            grip[j * XDIM + i] = [0, 0, 0, d, 0, 0, 0, -1, -1, -1]
                grip_now.append(grip.tolist())
    # print(err_n)
    return grip_now, vehicles_id


class InitMap2Grid:
    """
    将原始数据中的数据网格化，并保存起来，减少训练数据的时间
    """
    __constants__ = ['__grid_data', '__vehicles_id']

    def get_grid_from_map(self, time, vehicles):
        """
        获取当前时间路网下的所有网格化数据，按照车辆ID进行保存
        :return:
        """
        # print(json.dumps(vehicles))
        vehicles_id = []
        vehicles = vehicles['vehicles']
        grip_now = []
        # print(json.dumps(vehicles))
        err_n = 0
        for road_name in vehicles:
            # 如果车辆在RSG中而不是在路口中，则可以直接生成当前车辆的信息
            # 如果车辆是在路口中，则不加入，但是需要保留
            vehicles_in_road = vehicles[road_name]['vehicles']  # 获取在每个道路中的车辆情况

            if judge_road_junction(road_name):
                for vid in vehicles_in_road:
                    # 去除一些会对数据造成不利影响的数据
                    # 1. 到达道路尽头且到达终点的车辆
                    vehicle = vehicles_in_road[vid]
                    route = vehicle['info']['route']
                    target = route[-1]
                    position_lane = vehicle['info']['position_in_lane']
                    road_id = road_map.get_road_id(road_name)
                    # print(road_id)
                    # if road_map.get_road_length(road_id) - position_lane < 5:

                    # 遍历当前道路中的所有车辆
                    vehicles_id.append(vid)
                    grip = np.zeros((XDIM * YDIM, SDIM))

                    position = vehicle['info']['position']
                    lane_number = get_lane_number(vehicle['info']['lane'])
                    lateral_position = vehicle['info']['lateral_position']
                    # v_length = vehicle['info']['length']
                    v_length = 7.5

                    # 定义网格的边界
                    grip_x = [lane_number + 1, lane_number, lane_number - 1]
                    grip_y = [position_lane + (i - YDIM // 2) * v_length for i in range(YDIM)]

                    s_num = 0
                    for sid in vehicles_in_road:
                        if sid != vid:  # 去环境中找是否存在在网格中的车辆
                            s_vehicle = vehicles_in_road[sid]
                            s_lane = get_lane_number(s_vehicle['info']['lane'])
                            s_position_lane = s_vehicle['info']['position_in_lane']
                            s_position = s_vehicle['info']['position']
                            s_y = s_vehicle['info']['lateral_position']
                            if (grip_x[-1] <= s_lane <= grip_x[0]) \
                                    and (grip_y[0] <= s_position_lane <= grip_y[-1]):
                                s_num += 1
                                # 找出该车辆对应的网格
                                i = int(s_lane - grip_x[-1])
                                j = int((s_position_lane - grip_y[0]) / v_length)
                                # print(i, j)
                                # s_v = s_vehicle['info']['lateral_position']  # 斜向速度
                                s_h = s_vehicle['info']['speed']  # 水平速度
                                # a = s_vehicle['info']['accelerate']  # 水平加速度
                                d = math.sqrt(
                                    (s_position[-1] - position[-1]) ** 2 + (s_position[0] - position[0]) ** 2)  # 与主车的距离
                                angle = 3  # 下一个转向，除了本车之外转向都置为3
                                light = 2  # 交通信号灯，0为红灯，1为绿灯，2为其他车辆占位符
                                g_s = 1  # 网格的占位状态
                                g_d = road_map.get_road_length(road_id) - s_position_lane  # 网格与路口的距离
                                # 修改数据不用s_v, a
                                grip[j * XDIM + i] = [
                                    s_position_lane, s_y, s_h, d, angle, s_lane, light, g_s, road_id, g_d
                                ]

                    s_h = vehicle['info']['speed']
                    d = 0
                    # 预测车辆的下一个路口的转向状态
                    if road_id == road_map.get_road_id(target):
                        angle = 3
                    else:
                        angle = self.change_road.predict(np.array([[road_id, road_map.get_road_id(target)]]),
                                                         [time, ])
                        # print(road_id)
                        # print(road_map.get_road_id(target))
                        next_lane, junction = road_map.next_road(road_id, angle)

                    light = tls.get_current_state(time, junction.split('_')[0], junction) if angle != 3 else 2
                    g_s = 1
                    g_d = road_map.get_road_length(road_id) - position_lane
                    grip[XDIM * YDIM // 2] = [position_lane, lateral_position, s_h, d, angle, lane_number, light,
                                              g_s, road_id, g_d]
                    # print([position_lane, lateral_position, s, d, angle, lane_number, light, g_s, road_id, g_d])
                    road_length = road_map.get_road_length(road_id)
                    lane_num = road_map.get_lane_num(road_id)

                    # 将未有车辆占据的节点进行处理
                    for x in grip_x:
                        for y in grip_y:
                            i = int(x - grip_x[-1])
                            j = int((y - grip_y[0]) / v_length)
                            d = math.sqrt(
                                ((x - lane_number) * 3.2) ** 2 + (y - position_lane) ** 2
                            )  # 需要计算网格到目标车辆的距离
                            if np.sum(grip[j * XDIM + i]) == 0:
                                grip[j * XDIM + i] = [0, 0, 0, d, 0, x, 0, 0, road_id,
                                                      road_length - (position_lane + (j - YDIM // 2) * v_length)]
                            # 如果该网格不在道路内，则需要对其进行相应的处理
                            if x < 0 or x > (lane_num - 1) or y < 0 or y > road_length:
                                grip[j * XDIM + i] = [0, 0, 0, d, 0, 0, 0, -1, -1, -1]
                    grip_now.append(grip.tolist())
        # print(err_n)
        return grip_now, vehicles_id

    def __init__(
            self,
            vehicle_file,
            map_file,
            tls_file,
            traffic_file,
            load=True
    ):
        base_path = os.getcwd()
        self.load = load
        self.vehicle_file = os.path.join(base_path, vehicle_file)
        self.map_file = os.path.join(base_path, map_file)
        self.tls_file = os.path.join(base_path, tls_file)
        self.traffic_file = os.path.join(base_path, traffic_file)
        self.__grid_data = None
        self.__vehicles_id = None  # 每一秒的网格数据所属的车辆id
        self.change_road = ChangeRoadModel()
        if not self.load:
            self.map, self.vehicles, self.tls, self.traffic = self.__initialize_data()
        self.__get_current_data()

    def __initialize_data(self):
        with open(self.vehicle_file, 'r') as f:
            vehicles = json.load(f)

        with open(self.map_file, 'r') as f:
            maps = json.load(f)

        with open(self.tls_file, 'r') as f:
            tls = json.load(f)

        with open(self.traffic_file, 'r') as f:
            traffic = json.load(f)

        return maps, vehicles, tls, traffic

    def __get_current_data(self):
        if not self.load:  # 如果从未保存过那么保存
            length = len(self.vehicles)
            grip = []
            vehicles_id = []
            for i in range(500, 800):
                # if i != 0:
                #     return
                if i % 20 == 0:
                    print("当前已经加载{}s的数据".format(i))
                grid_now, vehicle_id = self.get_grid_from_map(i, self.vehicles[i])
                grip.append(grid_now)
                vehicles_id.append(vehicle_id)

            # 由于每一秒的数据维数不同，因此将其转换为numpy类型并不合适
            self.__grid_data = grip
            self.__vehicles_id = vehicles_id
            self.__save()
        else:
            with open('assets/grid.json', 'r') as f:
                # print(type(json.load(f)))
                res = json.loads(json.load(f))
                self.__grid_data, self.__vehicles_id = res['grid'], res['vid']

    def __save(self):
        res = json.dumps({
            'grid': self.__grid_data,
            'vid': self.__vehicles_id
        })
        with open('assets/grid.json', 'w') as f:
            json.dump(res, f)

    def get_grip(self):
        # s_position_lane, s_y, s_v, s_h, a, d, angle, s_lane, light, g_s, road_id, g_d
        return self.__grid_data[500:550], self.__vehicles_id[500:550], 500


if __name__ == '__main__':
    map2grip = InitMap2Grid(vehicles_file, road_map_file, traffic_light_file, roads_traffic_file, load=False)
    # res = map2grip.get_grip()
