"""
该部分用于处理交通信号灯数据
能够获取交通信号灯的状态
"""
import json
import os

from data_process.config import *


class TrafficLightSet:
    def __init__(
            self,
            tls_file=traffic_light_file
    ):
        self.tls_file = tls_file
        self.original_data = self.get_original_data()

    def get_original_data(self):
        filename = os.path.join(os.getcwd(), self.tls_file)
        with open(filename, 'r') as f:
            tls_data = json.load(f)
        return tls_data

    def get_current_state(self, time, intersection_name, intersection_lane):
        current_data = self.original_data[time]
        try:
            intersection_data = current_data[intersection_name]
        except KeyError:
            intersection_name = intersection_name[1:]
            intersection_data = current_data[intersection_name]
        tls_state = intersection_data['tls_state']
        controlled_lanes = intersection_data['controlled_links']
        for key, val in enumerate(controlled_lanes):
            if len(val) == 0:
                continue
            if intersection_lane in val[0]:
                state = tls_state[key]
        return 1 if state == 'G' or state == 'g' else 0


if __name__ == '__main__':
    tls = TrafficLightSet()
    light = tls.get_current_state(0, '-gneJ27', ':gneJ27_5_0')
    print(light)
