# here lies an example of how to train an RL agent


import argparse
import numpy as np
import os
from pathlib import Path
import sys

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
engine_path = os.path.join(base_dir, "olympics_engine")
sys.path.append(engine_path)

from env.chooseenv import make
import random

class random_agent:
    def __init__(self, seed=None):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]

    def get_action(self, observation):
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        return [[force], [angle]]

parser = argparse.ArgumentParser()
# parser.add_argument('--game_name', default="olympics-integrated", type=str)
parser.add_argument('--game_name', default="olympics-running", type=str)
parser.add_argument('--algo', default="ppo", type=str, help="ppo/sac")
parser.add_argument('--max_episodes', default=1500, type=int)
parser.add_argument('--episode_length', default=500, type=int)
parser.add_argument('--map', default=1, type = int)

parser.add_argument('--seed', default=1, type=int)

parser.add_argument("--save_interval", default=100, type=int)
parser.add_argument("--model_episode", default=0, type=int)

parser.add_argument("--load_model", action='store_true')
parser.add_argument("--load_run", default=2, type=int)
parser.add_argument("--load_episode", default=900, type=int)


device = 'cpu'
RENDER = True


def main(args):
    # build environment
    print(f"{args.game_name = }")
    env = make(env_type="olympics-integrated", game_name=args.game_name)
    print(f"{env = }")

    act_dim = env.action_dim
    obs_dim = 40*40
    # print(f'action dimension: {act_dim}')
    # print(f'observation dimension: {obs_dim}')

    agent = random_agent()

    num_episode = 0

    while num_episode < args.max_episodes:
        # rebuild each time to shuffle the running map
        env = make(env_type="olympics-integrated", game_name=args.game_name)
        state = env.reset()
        if RENDER:
            env.env_core.render()

        obs_ctrl_agent = np.array(state[0]['obs']['agent_obs'])
        obs_oppo_agent = np.array(state[1]['obs']['agent_obs'])

        num_episode += 1
        step = 0

        while True:
            action_ctrl = agent.get_action(obs_ctrl_agent)  # ctrl action
            # print(f'{obs_ctrl_agent.shape = }')
            # print(f'{action_ctrl = }')
            action_opponent = agent.get_action(obs_oppo_agent)  # opponent action
            # print(f'{obs_oppo_agent.shape = }')
            # print(f'{action_opponent = }')

            action = [action_ctrl, action_opponent]

            next_state, reward, done, _, info = env.step(action)

            step += 1

            obs_oppo_agent = np.array(next_state[0]['obs']['agent_obs'])
            obs_ctrl_agent = np.array(next_state[1]['obs']['agent_obs'])

            if RENDER:
                env.env_core.render()

            if done:
                break


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
