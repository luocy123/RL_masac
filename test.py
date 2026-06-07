import json
import math

import torch
from matplotlib import patches, pyplot as plt

from components.SAC import SAC
from components.config import load_config
from components.renderer_matplotlib import MatplotRenderer
from components.replaybuffer import ReplayBuffer
from env.env_test import World


seed = 42  #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_sac(model_path, cfg_path):
    cfg = load_config(cfg_path)
    env = World(cfg)
    states = env.reset()
    render = MatplotRenderer(env)
    state_dim = len(states[0])
    action_dim = 2
    sac = SAC(state_dim,action_dim)
    gamma = sac.gamma
    try:
        sac.load_model(model_path)
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"未找到模型文件: {model_path}")
        return

    # 训练参数
    test_cfg = cfg["test_cfg"]
    max_episodes = test_cfg["max_episodes"]
    max_steps = test_cfg["max_ticks"]
    replay_buffer_capacity = test_cfg["rb_capacity"]
    batch_size = test_cfg["batch_size"]
    # 初始化经验回放缓冲区

    arrival_counts = []
    for episode in range(max_episodes):
        states = env.reset()
        # 清空上一回合的轨迹数据
        render._uav_trajectories = {u.id: [] for u in env.uavs}
        render.get_states(states)  # 确保状态信息也被正确设置
        uav_num = cfg["red_uavs"]["uav_num"]
        episode_returns = [0] * uav_num

        discount = 1
        # render._uav_trajectories = {u.id: [] for u in env.uavs}
        for step_count in range(max_steps):
            actions = []
            for state in states:
                
                if len(state) == state_dim:
                    action = sac.select_action(state)
                    actions.append(action)
                elif len(state) == state_dim:
                    action = sac.select_action(state)
                    actions.append(action)
            render.get_states(states)
            next_states, rewards, dones = env.step(actions,states)

            render.draw()
            for i, reward, done in zip(range(uav_num), rewards, dones):
                if done:
                    episode_returns[i] += reward
                else:
                    episode_returns[i] += discount * reward
            discount *= gamma
            states = next_states

            if all(dones):
                break
        count = 0
        odom = 0
        total_arrival_time = 0
        if any(uav.arrival_flag for uav in env.uavs):
            for drone in env.uavs:
                if drone.arrival_flag:
                    count += 1
                    odom += drone.odometer
                    total_arrival_time += drone.arrival_time
            print(f"Episode: {episode + 1}")
            print(f"Total UAVSs:{uav_num}")
            print(f"Survivors: {count}")
            print(f"Survival rate: {count/uav_num}")
            print(f"Average path length: {odom/count}m")
            # print(f"average odom {odom/(count*env.W)}")
            print(f"Arrival elapsed time: {total_arrival_time/count}s")
            print(f"Total steps: {step_count}")
            arrival_count = {
                "arrival_count": count,
                "average_odom": odom/(count*env.W),
                "arrival_time": total_arrival_time/count
            }
            arrival_counts.append(arrival_count)
        print(f"Return = {episode_returns}")
    # 统计生存率大于0.5的回合数
    high_survival_episodes = sum(1 for item in arrival_counts if item["arrival_count"] / cfg["red_uavs"]["uav_num"] >= 0.5)
    percentage = (high_survival_episodes / max_episodes) * 100
    print(f"测试通过回合数比率:{percentage:.2f}%")

    # 计算所有测试回合的平均生存率
    total_survival_rate = sum(item["arrival_count"] / cfg["red_uavs"]["uav_num"] for item in arrival_counts)
    average_survival_rate = total_survival_rate / max_episodes if max_episodes > 0 else 0
    print(f"平均生存率: {average_survival_rate:.2%}")

    # 判断是否合格（平均生存率大于85%）
    if percentage > 0.5:
        print("测试结果: 合格")
    else:
        print("测试结果: 不合格")

    with open('data/arrival_counts1.json', 'w') as f:
        json.dump(arrival_counts, f)
    return env, sac
def main():
    cfg_path = "data/scenario2.json"
    model_path = "model/sac_model1.pth"
    test_sac(model_path, cfg_path)
if __name__ == '__main__':
    main()