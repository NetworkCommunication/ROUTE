"""
该程序用于道路的坐标转换，
将每一个道路的坐标进行变化，并保存参数
"""
import json
import math
import os
import numpy as np

from data_process.config import road_map_file, coordinate_file
from data_process.road_map import RoadMap
from data_process.tools import judge_road_junction
from data_process.vehicle_state import VehicleState


class CoordinateConversion:
    def __init__(self, road_map_file=road_map_file, coordinate_file=coordinate_file):
        self.road_map_file = road_map_file
        self.coordinate_file = coordinate_file
        self.road_map = RoadMap(road_map_file)

        if not os.path.isfile(self.coordinate_file):
            self.__save_coordinate()
        self.__coordinate_data = self.__load_coordinate()

    def __save_coordinate(self):
        with open(self.road_map_file, 'r') as f:
            map_data = json.load(f)

        coordinate_data = {}

        # 转换坐标，主要需要保存每一个道路的原始坐标、角度值、车道数量
        for road in map_data:
            road_id = road['road_id']
            if judge_road_junction(road_id):
                # 计算道路的角度
                shape = road['lane_links'][0]['lane_shape']
                p_1, p_3 = shape[0], shape[1]
                length = math.sqrt((p_3[1] - p_1[1]) ** 2 + (p_3[0] - p_1[0]) ** 2)
                theta = math.degrees(math.asin((p_3[1] - p_1[1]) / length))
                direct = p_3[0] - p_1[0]
                theta = 180 - theta if direct < 0 else theta  # 角度
                lane_num = road['lane_num']  # 道路中车道数
                lane_width = road['lane_links'][0]['width']  # 道路中车道的宽度

                coordinate_data[road_id] = {
                    'theta': theta,
                    'road_id': road_id,
                    'lane_num': lane_num,
                    'lane_width': lane_width,
                    'first_lane_shape': shape  # 0号车道的坐标
                }
                print(coordinate_data)
                # 保存坐标文件
                with open(self.coordinate_file, 'w') as f:
                    json.dump(coordinate_data, f)
                print("保存坐标文件完毕")

    def __load_coordinate(self):
        with open(self.coordinate_file, 'r') as f:
            data = json.load(f)
        return data

    def converse_one_coordinate(self, lane_num: int, position_in_lane: float, road_id: int):
        road_name = self.road_map.get_road_name(road_id)
        road_shape = self.__coordinate_data[road_name]
        x_hat = position_in_lane
        y_hat = lane_num * road_shape['lane_width']
        position_0 = road_shape['first_lane_shape']
        l = math.sqrt(x_hat ** 2 + y_hat ** 2)
        if l == 0:
            return position_0[0][0], position_0[0][1]
        alpha = math.degrees(math.asin(y_hat / l))
        degree = alpha + road_shape['theta']
        x = position_0[0][0] + l * math.cos(math.pi * degree / 180) - 1 * math.cos(math.pi * degree / 180)
        y = position_0[0][1] + l * math.sin(math.pi * degree / 180) - 1 * math.sin(math.pi * degree / 180)
        return x, y

    def converse_some_conversion(self, infos: np.ndarray):
        # 输入格式为[lane_name, position_in_lane, road_id]
        position = []
        for val in infos:
            lane_num, position_in_lane, road_id = val
            x, y = self.converse_one_coordinate(int(lane_num), position_in_lane, int(road_id))
            position.append([x, y])
        return position


if __name__ == '__main__':
    coordinate_conversion = CoordinateConversion()
    vehicle_state = VehicleState()
    # x, y, road_id, lane_num, position_lane, a, v_s, h_s
    res = vehicle_state.get_some_vehicles_state(5, [('1', ), ('5', ), ('2', )])
    info = np.zeros((len(res), 3))
    info[:, :2] = np.array(res)[:, 3:5]
    info[:, 2] = np.array(res)[:, 2]
    print(info)
    t = coordinate_conversion.converse_some_conversion(info)
    print(t)
    print(res)
