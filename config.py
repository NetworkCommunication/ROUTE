import os
# 设置训练数据的车道
train_road = 'gneE24'

# 设置测试数据的车道
test_road = '-gneE30'

#  json_roads_traffic = json.dumps(self.roads_info)
#  json_road_map = json.dumps(self.road_map)
#  json_traffic_light = json.dumps(self.traffic_light)
#  json_vehicles = json.dumps(self.vehicles)

# 交通流文件名

base_path = os.getcwd()
road_map_file = 'assets/road_map.json'
road_map_file = os.path.join(base_path, road_map_file)

roads_traffic_file = 'assets/roads_traffic.json'
traffic_light_file = 'assets/traffic_light.json'
vehicles_file = 'assets/vehicles.json'
coordinate_file = 'assets/coordinate.json'
grid_temporal_cache = 'grid_temporal_cache.json'
