import os
import sys
from pathlib import Path
project_path = str(Path(__file__).resolve().parent.parent)

# Add ppo module in system path and project path
sys.path.append("AI_Application_Practice/06.PPO/")
sys.path.append(project_path)

from b_ppo import PPOAgent
import numpy as np
import torch
import random

import gymnasium as gym

from d_wrappers import CompetitionOlympicsEnvWrapper
import argparse
from config import args_olympic_running, args_olympic_wrestling, args_olympic_integrated

from termproject_olympic.env.chooseenv import make

DEVICE = None

class GlobalConfig:
    def __init__(self):
        self.seed = 555
        self.path2save_train_history = "train_history"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

config = GlobalConfig()
seed_everything(config.seed)


def make_env(env_name, agent=None, config=None):
    # environment
    env = make(env_type="olympics-integrated", game_name=env_name)
    if config.smart_competition:
        env = CompetitionOlympicsEnvWrapper(env, agent=agent, args=config)
    else:
        env = CompetitionOlympicsEnvWrapper(env, args=config)

    return env


def main(args, evaluation=False):
    if not evaluation:
        # ppo agent
        agent = PPOAgent(make_env, args)
        # ppo train
        agent.train()
    else:
        agent = PPOAgent(make_env, args)
        agent.load_predtrain_model(f"{args.path2save_train_history}/actor.pth", f"{args.path2save_train_history}/critic.pth")
        agent.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", default="olympics-running")
    # "CartPole-v1", "Pendulum-v1", "Acrobot-v1", "LunarLanderContinuous-v2"
    # "LunarLander-v2", "BipedalWalker-v3", "MountainCarContinuous-v0"
    args, rest_args = parser.parse_known_args()
    env_name = args.env_type

    if env_name == "olympics-running":
        args = args_olympic_running.get_args(rest_args)
    elif env_name == "olympics-wrestling":
        args = args_olympic_wrestling.get_args(rest_args)
    elif env_name == "olympics-integrated":
        args = args_olympic_integrated.get_args(rest_args)
    else:
        raise Exception("Invalid Environment")

    args.path2save_train_history = config.path2save_train_history

    if not os.path.exists(args.path2save_train_history):
        try:
            os.mkdir(args.path2save_train_history)
        except:
            dir_path_head, dir_path_tail = os.path.split(args.path2save_train_history)
            if len(dir_path_tail) == 0:
                dir_path_head, dir_path_tail = os.path.split(dir_path_head)
            os.mkdir(dir_path_head)
            os.mkdir(args.path2save_train_history)

    main(args, args.is_evaluate)
