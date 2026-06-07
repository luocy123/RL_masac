import json
import math

import torch
from matplotlib import patches, pyplot as plt

from components.SAC import SAC
from components.config import load_config
from components.renderer_matplotlib import MatplotRenderer
from components.replaybuffer import ReplayBuffer
from env.env1 import World


seed = 42  #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_sac(model_path, cfg_path):
    cfg = load_config(cfg_path)
    env = World(cfg)
    states = env.reset()
    render = MatplotRenderer(env)
    state_dim = len(states[0])
    action_dim = 2
    sac = SAC(state_dim,action_dim)
    gamma = sac.gamma
    if model_path is not None:
        try:
            sac.load_model(model_path)
            print(f"成功加载模型: {model_path}")
        except FileNotFoundError:
            print(f"未找到领导者模型文件: {model_path}，将从头开始训练。")
    else:
        print("未提供领导者模型文件，将从头开始训练。")

    # 训练参数
    train_cfg = cfg["train_cfg"]
    max_episodes = train_cfg["max_episodes"]
    max_steps = train_cfg["max_ticks"]
    replay_buffer_capacity = train_cfg["rb_capacity"]
    batch_size = train_cfg["batch_size"]
    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
    # 训练循环
    all_episode_data = []  # 用于保存所有 episode 的数据

    for episode in range(max_episodes):
        states = env.reset()
        uav_num = cfg["red_uavs"]["uav_num"]
        episode_returns = [0] * uav_num

        discount = 1
        episode_path = []
        episode_path.extend(env.start_pos)  # 保存当前 episode 小车的轨迹
        for step_count in range(max_steps):
            actions = []
            for state in states:
                action = sac.select_action(state)
                actions.append(action)
            # render.get_states(states)
            next_states, rewards, dones = env.step(actions, states)
            # render.get_rewards(rewards)
            # render.draw()
            for i, state, action, reward, next_state, done in zip(range(uav_num), states, actions, rewards,
                                                                  next_states, dones):

                replay_buffer.add(state, action, reward, next_state, done)

                if done:
                    episode_returns[i] += reward
                else:
                    episode_returns[i] += discount * reward
            discount *= gamma
            states = next_states

            uavs = env.uavs
            for i in range(uav_num):
                x, y = uavs[i].position
                episode_path.append([x, y])
            if step_count % 4 == 0:
                if len(replay_buffer) >= batch_size:
                    batch = replay_buffer.sample(batch_size)
                    sac.update(batch)

            if all(dones):
                break
        # 保存当前 episode 的数据

        episode_data = {
            "episode": episode,
            "path": episode_path,
            "return": episode_returns,
            "q1_loss_leader": sac.q1_loss,
            "q2_loss_leader": sac.q2_loss,
            "actor_loss_leader": sac.actor_loss,
            "a_loss_leader": sac.a_loss
        }
        all_episode_data.append(episode_data)
        count = 0
        if any(uav.arrival_flag for uav in env.uavs):
            for drone in env.uavs:
                if drone.arrival_flag:
                    count += 1
            print(f"survive follow drone count {count}")
        print(f"Episode {episode + 1}: Return = {episode_returns}")

    # 保存所有 episode 数据到 JSON 文件
    with open('data/episode_data100.json', 'w') as f:
        json.dump(all_episode_data, f)

    sac.save_model(f'model/sac_model100.pth')

    return env, sac
def main():
    cfg_path = "data/scenario.json"
    model_path = "model/sac_model100.pth"
    train_sac(model_path, cfg_path)

if __name__ == '__main__':
    main()