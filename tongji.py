import json
import math
import chardet
import numpy as np
from matplotlib import pyplot as plt, patches

def plot_arrival_data(file_path):
    with open(file_path, 'r') as f:
        arrival_datas = json.load(f)
    arrival_count = []  
    for arrival_data in arrival_datas:
        arrival_count.append(arrival_data['arrival_count'])

    plt.plot(list(range(len(arrival_count))), arrival_count)
    print(f"average arrival count {sum(arrival_count)/len(arrival_count)}")

if __name__ == '__main__':
    plot_arrival_data("data/arrival_counts1.json")