# ROUTE

The purpose of this project is to realize a precise urban regional vehicle trajectory prediction, which can also be used to predict the future traffic flow by the integral vehicle's future trajectory. This project uses transformer-GAN based method to predict the future trajectory of vehicles in each road segment, the DNN is used to predict the future turning of vehicles that need to change road segment. In addition, the coordinate conversion model is used to simplify the influence of complexity of the spatial of the regional road network. Finally, ROUTE can predict the future trajectory of each vehicle in the urban regional road network by iteratively analyzing the microstate of each vehicle.

## Instructions:

1. The file 'model' is used to define the vehicle trajectory prediction model.
2. The file 'coordinate_conversion' is used to converse the coordinate of each vehicle.
3. The file 'config' is used to define the data filename.
4. The file 'road_map' is used to get complex road topology.
5. The file 'grid_map' is used to grid the surrounding state of traget vehicle.
6. The file 'traffic_light' is used to get traffic light state.
7. The file 'vehicle_state' is used to get the real vehicle state for training and verifing model.
8. The file 'change_road_model' is used to define the change road model.
9. The file 'datasets' and 'dataset_choice_road' are used to deal with the data.
10. The file 'route' contains model training and verfing of ROUTE.
