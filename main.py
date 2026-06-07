import numpy as np
from config import load_config
from world import World
from renderer_matplotlib import MatplotRenderer
import time
from logger_csv import write_sim_csv
from logger_excel import write_sim_xlsx
import matplotlib.pyplot as plt
def main():
    cfg = load_config("../data/scenario.json")
    world = World(cfg)
    view = MatplotRenderer(world)

    # 简单实时循环（matplotlib）
    while not world.done():
        world.step()
        view.draw()
    # 仿真结束 → 写CSV（文件名含日期+时间+仿真名）
    write_sim_csv(world, sim_name="scenario")
    write_sim_xlsx(world, sim_name="scenario")
    print("Simulation finished.")
    plt.show()
if __name__ == "__main__":
    main()
