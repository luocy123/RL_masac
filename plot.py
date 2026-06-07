import json
import math
import chardet
import numpy as np
from matplotlib import pyplot as plt, patches
# 检测文件编码


def plot_episode_data(episode_num=None):
    """
    绘制指定 episode 或所有 episode 的轨迹数据
    :param episode_num: 指定要绘制的 episode 编号，若为 None 则绘制所有 episode
    """
    try:
        with open('data/episode_data_static1.json', 'r') as f:
            all_episode_data = json.load(f)

        if episode_num is not None:
            if 0 <= episode_num < len(all_episode_data):
                episode_data = all_episode_data[episode_num]
                plot_single_episode_more(episode_data, 100)
                # plot_single_episode(episode_data)
            else:
                print(f"指定的 episode 编号 {episode_num} 超出范围。")
        else:
            for episode_data in all_episode_data:
                plot_single_episode_more(episode_data)
                # plot_single_episode(episode_data)
    except FileNotFoundError:
        print("未找到 episode 数据文件，请先运行训练函数。")

def plot_single_episode_more(episode_data,drone_count=1):
    """
    绘制单个 episode 的轨迹数据
    :param episode_data: 单个 episode 的数据
    """
    episode = episode_data["episode"]
    start_positions = episode_data["start_position"]
    target_x, target_y = episode_data["target_position"]
    obstacle_circles = episode_data["obstacle_circles"]

    paths = episode_data["path"]

    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')

    # 绘制障碍物

    for circle_x, circle_y, r in obstacle_circles:
        circle = patches.Circle((circle_x, circle_y), r, edgecolor='black', facecolor='black')
        ax.add_patch(circle)


    target_circle = patches.Circle((target_x, target_y), 5, edgecolor='blue', facecolor='none')
    ax.add_patch(target_circle)

    # 绘制起始位置（绿色点）
    for pos in start_positions:
        start_x, start_y = pos
        ax.plot(start_x, start_y, 'go', label='Start')

    # 绘制路径
    for i in range(drone_count):
        path = paths[i::drone_count]
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'r-', label='Path')

    ax.legend()
    plt.title(f"Episode {episode}: Reward = {episode_data['return']}")
    plt.show()
    # plt.show(block=False)
    #
    # plt.pause(1)
    #
    # plt.close()

def plot_loss_data(file_path1):
    try:
        episodes = []
        q1_loss = []
        q2_loss = []
        actor_loss = []
        a_loss = []
        with open(file_path1, 'r') as f:
            episode_data = json.load(f)

        # episode_data = episode_data[800: ]
        for episode in episode_data:
            episodes.append(episode['episode'])
            q1_loss.append(episode['q1_loss_leader'])
            q2_loss.append(episode['q2_loss_leader'])
            actor_loss.append(episode['actor_loss_leader'])
            a_loss.append(episode['a_loss_leader'])

        # with open(file_path2, 'r') as f:
        #     episode_data = json.load(f)

        # episode_data = episode_data[800: ]
        # for episode in episode_data:
        #     episodes.append(episode['episode']+1000)
        #     q1_loss.append(episode['q1_loss_leader'])
        #     q2_loss.append(episode['q2_loss_leader'])
        #     actor_loss.append(episode['actor_loss_leader'])
        #     a_loss.append(episode['a_loss_leader'])

        plt.figure(figsize=(10, 6))
        # plt.plot(episodes, q1_loss, label='Q1 loss per Episode')
        # plt.plot(episodes, q2_loss, label='Q2 loss per Episode')
        plt.plot(episodes, actor_loss, label=' policy loss per Episode')
        # plt.plot(episodes, a_loss, label='alpha loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('loss per Episode')
        plt.legend()
        plt.grid(True)
        plt.show()
    except FileNotFoundError:
        print(f"错误: 文件 {file_path1} 未找到。")
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path1} 不是有效的 JSON 文件。")
    except KeyError:
        print("错误: JSON 文件中缺少 'episode' 或 'loss' 键。")

def plot_reward_data(file_path1,file_path2):

    episodes = []
    rewards = []
    with open(file_path1, 'r') as f:
        episode_data = json.load(f)
    for episode in episode_data:
        episodes.append(episode['episode'])
        rewards.append(episode['return'])
    # with open(file_path2, 'r') as f:
    #     episode_data = json.load(f)
    # for episode in episode_data:
    #     episodes.append(episode['episode']+1000)
    #     rewards.append(episode['return'])
    Return = [sum(reward)/len(reward) for reward in rewards]
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, Return, label=' Average reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    data_path1 = "data/episode_data5.json"
    data_path2 = "data/episode_data_dy8_random_1.json"
    model_path = "model/sac_model_static_L.pth"
    plot_loss_data(data_path1)
    # plot_episode_data(1978)
    # for i in range(100):
    #     plot_episode_data(i+900)