"""
存储道路文件
"""
import json
import os

import numpy as np

from data_process.config import road_map_file
from data_process.tools import judge_road_junction, get_road_name, get_junction_name


class RoadMap:
    """
    处理路网数据，将路网数据变成便于使用的数据格式
    """

    def __init__(
            self,
            map_file=road_map_file  # 存储地图数据的文件
    ):
        self.map_file = map_file
        self.original_data = self.get_original_data()
        self.roads, self.v_num = self.make_order()
        self.junctions, self.node_num = self.make_node_order()
        self.map = np.full((self.node_num, self.node_num), fill_value=0)
        self.map_index = self.build_adjacency_list()
        self.index_id = self.build_adjacency_list()
        self.light_map = np.zeros((self.v_num, self.v_num))
        # print(self.res)
        self.direction = ['r', 's', 'l']
        self.direction_id = {'r': 0, 's': 1, 'l': 2}

    def get_index_by_id(self, rid):
        # 返回矩阵中的索引
        for index in self.map_index:
            if self.map_index[index] == rid:
                return index

    def get_id_by_index(self, index):
        return self.map[index]

    def get_original_data(self):
        filename = os.path.join(os.getcwd(), self.map_file)
        with open(filename, 'r') as f:
            map_data = json.load(f)
        return map_data

    def make_order(self):
        # 将每一个道路给予一个编号
        roads = []
        for road in self.original_data:
            if judge_road_junction(road['road_id']):
                roads.append(road['road_id'])
        return roads, len(roads)

    def make_node_order(self):
        junctions = [
            ':gneJ33', ':gneJ28', ':gneJ25',
            ':gneJ32', ':gneJ27', ':gneJ23',
            ':gneJ34', ':gneJ29', ':gneJ30'
        ]
        # for road in self.original_data:
        #     if not judge_road_junction(road['road_id']):
        #         junctions.append(get_junction_name(road['road_id']))
        #         print(get_junction_name(road['road_id']))
        # junctions = list(set(junctions))
        return junctions, len(junctions)

    def get_junction_id(self, junction_name):
        return self.junctions.index(junction_name)

    def get_junction_name_by_id(self, junction_id):
        return self.junctions[int(junction_id)]

    def get_junction_between_two_road(self, road_1, road_2):
        jun = -1
        link_1 = None
        link_2 = None
        for key in self.index_id:
            if self.index_id[key] == road_1:
                link_1 = key
            if self.index_id[key] == road_2:
                link_2 = key

        if link_1 is not None and link_2 is not None:
            for v in link_1:
                if v in link_2:
                    jun = v

        return jun

    def build_adjacency_list(self):
        links = {}
        rsg_links = {}
        for val in self.original_data:
            road_name = val['road_id']
            # 如果不是道路，而是交叉路口，则处理
            if judge_road_junction(road_name):
                junction_name = get_junction_name(road_name)
                if junction_name not in links:
                    links[junction_name] = []
                next_links = val['lane_links']

                for road in next_links:
                    links[junction_name].append(get_road_name(road['next_lane_id']))
                    rsg_links[junction_name] = get_junction_name(road['next_junction_id'])

        # print(rsg_links)
        # print(links)
        map_index = {}

        for k_1 in rsg_links:
            jun_name = rsg_links[k_1]
            jun_id = self.get_junction_id(jun_name)

            road_list = links[k_1]
            for road in road_list:
                next_jun = rsg_links[road]
                next_jun_id = self.get_junction_id(next_jun)
                road_id = self.get_road_id(road)
                map_index[(jun_id, next_jun_id)] = road_id

        # print(map_index)
        return map_index

    def next_direction(self, road_name, next_road):
        # 获取去往下一条道路的方向
        if road_name == next_road:
            return 3
        for road in self.original_data:
            if road_name == road['road_id']:
                for link in road['lane_links']:
                    if link['next_lane_id'].split('_')[0] == next_road:
                        return self.direction_id[link['next_turn_direction']]

    def next_direction_by_id(self, rid, nid):
        r_name = self.get_road_name(int(rid))
        n_name = self.get_road_name(int(nid))
        return self.next_direction(r_name, n_name)

    def get_road_id(self, road):
        # 获得道路的编号
        return self.roads.index(road)

    def get_road_name(self, rid: int):
        # 获得道路编号对应的道路名
        return self.roads[rid]

    def get_road_length(self, rid):
        # 获取道路的长度
        road_name = self.roads[rid]
        length = 0
        for road in self.original_data:
            if road['road_id'] == road_name:
                length = road['length']
        return length

    def get_lane_num(self, rid):
        # 获取道路中车道的数量
        road_name = self.roads[rid]
        number = 0
        for road in self.original_data:
            if road['road_id'] == road_name:
                number = road['lane_num']
        return number

    def next_road(self, rid, direction):
        # 获取下一个转向方向的情况
        road_name = self.roads[rid]
        if direction == 3:
            return road_name, None
        for road in self.original_data:
            if road['road_id'] == road_name:
                lane_links = road['lane_links']
                for link in lane_links:
                    if link['next_turn_direction'] == self.direction[direction]:
                        return link['lane_id'], link['next_junction_id']
        return road_name, None

    def get_nex_road(self, rid, direction):
        # 获取下一条道路
        road_name = self.roads[rid]
        if direction == 3:
            return road_name
        for road in self.original_data:
            if road['road_id'] == road_name:
                lane_links = road['lane_links']
                for link in lane_links:
                    if link['next_turn_direction'] == self.direction[direction]:
                        return link['next_lane_id'].split('_')[0]

    def get_road_shape(self, rid):
        # 获取当前道路的形状
        road_name = self.roads[rid]

    def get_neighbor(self, rid):
        nei = [0 for _ in range(self.v_num)]
        r_map_index = self.map_index
        k = dict(zip(r_map_index.values(), r_map_index.keys()))
        # 找到以自己的尾端交叉路口为起始的道路
        inter = k[rid][1]  # (0, 1)
        for val in r_map_index:
            # print(val[0])
            # print(inter)
            if inter == val[0] and k[rid][0] != val[1]:
                # road_name = self.get_road_name(rid).split('-')[1]
                # next_road_name = self.get_road_name(r_map_index[val]).split('-')[1]
                #
                # if road_name != next_road_name:
                nei[r_map_index[val]] = 1
        return nei

    def get_many_neighbor(self, rids):
        res = []
        for rid in rids:
            res.append(self.get_neighbor(int(rid)))
        return res


if __name__ == '__main__':
    road_map = RoadMap()
    print(road_map.get_neighbor(12))
    # print(road_map.get_road_name(1))
