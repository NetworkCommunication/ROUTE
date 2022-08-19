import json

from data_process.config import vehicles_file
from data_process.road_map import RoadMap
from data_process.tools import judge_road_junction, get_lane_number



class VehicleState:
    """
    存储每一秒的车辆状态信息。
    该类提供接口：
    1. 返回指定秒所有车辆的状态信息:
        x, y, road_id, lane_num, position_in_lane, v_s, a, h_s
    2. 返回车辆的起始地点和终点
    """

    def __init__(self, vehicle_file=vehicles_file, load=True):
        self.vehicle_file = vehicle_file
        self.load = load
        self.vehicle_state = None
        self.vehicle_trip = None
        if not load:
            self.vehicles = self.__initialize_data()
        self.__get_current_data()

    def __initialize_data(self):
        with open(self.vehicle_file, 'r') as f:
            vehicles = json.load(f)

        return vehicles

    def __get_current_data(self):
        """
        获取每秒的车辆状态信息，需要将原始的按照道路展开的数据首先按照时间展开，然后在按照车辆展开
        :return:
        """
        if not self.load:
            v_state = []
            v_trip = {}
            # print(self.vehicles[100])
            for v in self.vehicles:
                # time = int(v['time'])  # 时间
                vehicles = v['vehicles']
                s_per = {}
                for road_name in vehicles:  # 获取每条道路中的车辆
                    # 如果道路存在id则获取id，如果是交叉路口那么直接使用交叉路口名
                    road_id = road_map.get_road_id(road_name) if judge_road_junction(road_name) else road_name
                    r = vehicles[road_name]['vehicles']
                    for vid in r:  # 获取每个车辆的信息
                        vehicle = r[vid]
                        v_trip[vid] = vehicle['info']['route']
                        x, y = vehicle['info']['position']
                        position_in_lane = vehicle['info']['position_in_lane']
                        v_s = vehicle['info']['speed']
                        h_s = vehicle['info']['lateral_speed']
                        lane_num = get_lane_number(vehicle['info']['lane'])
                        a = vehicle['info']['accelerate']
                        s_per[vid] = [x, y, road_id, lane_num, round(position_in_lane, 2), round(v_s, 2),
                                      round(a, 2), round(h_s, 2)]
                v_state.append(s_per)
            self.vehicle_state = v_state
            with open('assets/vehicle_state.json', 'w') as f:
                json.dump(json.dumps(v_state), f)
            with open('assets/trip.json', 'w') as f:
                json.dump(json.dumps(v_trip), f)
        else:
            with open('assets/vehicle_state.json', 'r') as f:
                v_state = json.loads(json.load(f))
                self.vehicle_state = v_state
            with open('assets/trip.json', 'r') as f:
                v_trip = json.loads(json.load(f))
                self.vehicle_trip = v_trip

    def get_one_vehicle_state(self, t: int, vid: str) -> list:
        """获取一辆车的状态"""
        return self.vehicle_state[t][vid]

    def get_some_vehicles_state(self, t: int, vids: list) -> list:
        """
        get some vehicles' state
        :param t: time
        :param vids: vehicles id list
        :return states: vehicles state
        """
        states = []
        for vid in vids:
            try:  # 如果找不到目标的话，说明该车已经到达的目的地，那么返回0
                # 如果车辆进入交叉路口，那么返回进入交叉路口前的位址, 道路编号置为-1
                uni = self.get_one_vehicle_state(t - 1, vid) if type(
                    self.get_one_vehicle_state(t, vid)[2]) == str else self.vehicle_state[t][vid]
                res = [i for i in uni]
                if uni[2] == -1:
                    print(t)
                    print(vid)
                    print(self.get_one_vehicle_state(t, vid))

                if type(self.vehicle_state[t][vid][2]) is not int:
                    res[2] = -1
                # res[2] = res[2] if type(
                #     self.get_one_vehicle_state(t, vid)[2]) == int else -1

                res[4] = res[4] if res[2] != -1 else 500
                # res[4] = res[4] if type(
                #     self.get_one_vehicle_state(t, vid)[2]) != -1 else 500
                states.append(res)
            except KeyError:
                res = [0, 0, -2, 0, 500, 0, 0, 0]
                # res = self.vehicle_state[t - 1][vid]
                # res[2] = -2
                states.append(res)
        return states

    def get_vehicle_trip(self, vid: str) -> list:
        """获取车辆的起始和终点"""
        return self.vehicle_trip[vid]


if __name__ == '__main__':
    road_map = RoadMap()
    vehicle_state = VehicleState(load=True)
    print(vehicle_state.get_one_vehicle_state(501, '575'))
    # print(vehicle_state.get_vehicle_trip('62'))