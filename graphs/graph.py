import numpy as np
import json
from pprint import pprint

with open('gurnee/gurnee-graph.json', 'r') as f:
    graph = json.load(f)




adjacency = np.array(graph['adjacency-matrix'])
distance = np.array(graph['distance-matrix'])
print(distance)
print("---------------")
print(adjacency)
sensor_dictionary = graph['sensor-dictionary']

pprint(sensor_dictionary['28'])
pprint(sensor_dictionary['56'])
pprint(sensor_dictionary['84'])

'''
相机 经纬度  相机朝向  入度还是出度
[[42.363587, -87.929395], 'IL 21 at Washington East-inbound']
[[42.377277, -87.904064], 'US 41 at Delany East-inbound']
[[42.393079, -87.926813], 'US 41 at Stearns School North-inbound']
'''

'''
相机位置信息   
邻接矩阵信息  318 * 318  0 1 1.5 标记什么意思
距离信息    318 * 318
'''
