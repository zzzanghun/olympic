import gym
import numpy
import numpy as np
import random
import torch

from collections import deque
import re

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

GAME_NAME_LIST = [
    'running-competition', 'wrestling', 'football', 'table-hockey'
]


class DummyCompetitionOlympicsAgent:
    def __init__(self):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]

    def seed(self, seed=None):
        random.seed(seed)

    def get_action(self, obs):
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        return [[force], [angle]]


class CompetitionOlympicsEnvWrapper(gym.Wrapper):
    metadata = {}

    def __init__(self, env, agent=None, args=None):
        super().__init__(env)

        self.args = args
        self.controlled_agent_index = args.controlled_agent_index

        self.args = args
        assert self.args

        self.frame_stack = args.frame_stack
        assert self.frame_stack > 0 or isinstance(self.frame_stack, int)
        shape = [self.frame_stack, 40, 40]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=shape, dtype=np.float64
        )
        self.frames_controlled = deque([], maxlen=self.frame_stack)
        self.frames_opponent = deque([], maxlen=self.frame_stack)

        self.action_space = gym.spaces.Box(
            low=np.asarray([-100., -30.0]), high=np.asarray([200., 30.0]), shape=(2,), dtype=float
        )

        self.sub_game = self.args.env_name

        self.smart_competition = args.smart_competition
        if self.smart_competition:
            self.agent = agent

        self.opponent_agent = DummyCompetitionOlympicsAgent()
        self.last_observation_opponent_agent = None

        self.running_arrow = False
        self.running_step_for_reward = 0
        self.goal_line_running_action = [[200.], [0.]]
        self.goal_line_running_flag = False
        self.episode_steps = 0
        self.total_steps = 0

        self.device = args.device

        self.frame_skipping = args.frame_skipping
        self.last_action = None

    def _get_normalize_observation(self, observation):
        return observation / 10.0

    def reset(self, return_info=True):
        self.episode_steps = 0
        observation = self.env.reset()

        observation_opponent_agent = self.convert_obs_opponent_to_controlled(
            observation[1 - self.controlled_agent_index]['obs']['agent_obs']
        )
        observation_opponent_agent = np.expand_dims(
            observation_opponent_agent, axis=0
        )
        observation_controlled_agent = np.expand_dims(
            observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0
        )

        # print(list(observation_controlled_agent[0][-2]) == list(self.wrestling_reset_observation),
        #       list(observation_controlled_agent[0][-2]) == list(self.football_reset_observation),
        #       list(observation_controlled_agent[0][-2]) == list(self.table_hockey_reset_observation_0),
        #       list(observation_controlled_agent[0][-2]) == list(self.table_hockey_reset_observation_1),
        #       self.env.env_core.current_game.game_name, "!!!!!!!!!!!!!!!!!!")

        observation_controlled_agent = self._get_normalize_observation(observation_controlled_agent)
        ######### frame stack #########
        for _ in range(self.frame_stack):
            self.frames_controlled.append(observation_controlled_agent)
        observation_controlled_agent = self._transform_observation(self.frames_controlled)
        ################################

        observation_opponent_agent = self._get_normalize_observation(observation_opponent_agent)
        ######### frame stack #########
        for _ in range(self.frame_stack):
            self.frames_opponent.append(observation_opponent_agent)
        observation_opponent_agent = self._transform_observation(self.frames_opponent)
        ################################

        self.last_observation_opponent_agent = observation_opponent_agent

        # print(f"{observation_controlled_agent.shape = }, !!!!!!!!!!!")

        if return_info:
            info = {"game_name": self.sub_game}

            return [observation_controlled_agent], info
        else:
            return [observation_controlled_agent]

    def step(self, action_controlled):
        self.episode_steps += 1
        self.total_steps += 1

        action_controlled = self.get_scaled_action(action_controlled)

        if self.args.render_over_train or self.args.is_evaluate:
            self.render()

        # action_controlled = np.expand_dims(action_controlled[0], axis=1)
        # action_opponent = self.opponent_agent.get_action(self.last_observation_opponent_agent)
        # action = [action_opponent, action_controlled] if self.controlled_agent_index == 1 else [action_controlled,
        # 																						action_opponent]
        # action = [[[0.],[0.]], [[50.], [0.]]]
        # action = [[[50.], [0.]], [[50.], [30.]]]
        # if self.goal_line_running_flag:
        # 	action[self.controlled_agent_index] = self.goal_line_running_action
        # 	action[1 - self.controlled_agent_index] = action[1 - self.controlled_agent_index]

        action_opponent = self.get_action_opponent(self.last_observation_opponent_agent)
        action_controlled = np.expand_dims(action_controlled, axis=1)
        action_opponent = np.expand_dims(action_opponent, axis=1)
        action = [action_opponent, action_controlled] if self.args.controlled_agent_index == 1 else [
            action_controlled, action_opponent]

        for i in range(self.frame_skipping):
            next_observation, reward, done, info_before, info_after = self.env.step(action)
            if done:
                break
        # if next_observation[0]['obs']['game_mode'] == 'NEW GAME':
        # 	if list(next_observation[0]['obs']['agent_obs'][-2]) == list(self.wrestling_reset_observation):
        # 		self.sub_game = 'wrestling'
        # 	elif list(next_observation[0]['obs']['agent_obs'][-2]) == list(self.football_reset_observation):
        # 		self.sub_game = 'football'
        # 	elif list(next_observation[0]['obs']['agent_obs'][-2]) == list(self.table_hockey_reset_observation_0) or \
        # 			list(next_observation[0]['obs']['agent_obs'][-2]) == list(self.table_hockey_reset_observation_1):
        # 		self.sub_game = 'table-hockey'
        # 	else:
        # 		self.sub_game = 'running-competition'
        # 	assert self.sub_game == self.env.env_core.current_game.game_name
        next_observation_opponent_agent = self.convert_obs_opponent_to_controlled(
            next_observation[1 - self.controlled_agent_index]['obs']['agent_obs']
        )
        next_observation_opponent_agent = np.expand_dims(
            next_observation_opponent_agent, axis=0
        )
        next_observation_controlled_agent = np.expand_dims(
            next_observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0
        )

        next_observation_controlled_agent = self._get_normalize_observation(next_observation_controlled_agent)
        ######### frame stack #########
        self.frames_controlled.append(next_observation_controlled_agent)
        next_observation_controlled_agent = self._transform_observation(self.frames_controlled)
        ################################

        next_observation_opponent_agent = self._get_normalize_observation(next_observation_opponent_agent)
        ######### frame stack #########
        self.frames_opponent.append(next_observation_opponent_agent)
        next_observation_opponent_agent = self._transform_observation(self.frames_opponent)
        self.last_observation_opponent_agent = next_observation_opponent_agent
        ################################

        reward_controlled = reward[self.controlled_agent_index]
        reward_opponent = reward[1 - self.controlled_agent_index]

        info = {
            "game_name": self.sub_game,
            "original_reward": reward,
            "game_mode": next_observation[0]['obs']['game_mode']
        }

        # if done:
        # 	if reward[self.controlled_agent_index] > reward[1 - self.controlled_agent_index]:
        # 		info['win_controlled_agent'] = True
        # 	elif reward[self.controlled_agent_index] < reward[1 - self.controlled_agent_index]:
        # 		info['win_opponent_agent'] = True
        # 	elif reward[self.controlled_agent_index] == reward[1 - self.controlled_agent_index]:
        # 		info['draw'] = True
        # 	else:
        # 		raise ValueError()

        if self.sub_game == "olympics-wrestling":
            if done:
                if not reward_controlled == 100:
                    reward_controlled = 10
                else:
                    reward_controlled = -10
            reward_shaped = self.wrestling_reward(next_observation[self.controlled_agent_index]['obs']['agent_obs'])
            reward_controlled += reward_shaped
        elif self.sub_game == "olympics-running":
            if done:
                if not reward_controlled == 100:
                    reward_controlled = 10
                else:
                    reward_controlled = -10
            reward_shaped = self.running_reward(
                next_observation[self.controlled_agent_index]['obs']['agent_obs'],
                next_observation[self.controlled_agent_index]['obs']['energy']
            )
            reward_controlled += reward_shaped
        elif self.sub_game == "olympics-integrated":
            pass
        else:
            raise ValueError()
        return [next_observation_controlled_agent], reward_controlled, done, False, info

    def _get_reward_shaped(self, reward, done, controlled_agent_index):
        if done and reward[0] != reward[1]:
            reward_shaped = [0.0, 10.0] if reward[0] < reward[1] else [10.0, 0.0]
        # print("-" * 10, reward, reward_shaped, reward_shaped[controlled_agent_index])
        else:
            # if reward[0] != reward[1]:
            # 	print("-" * 10, reward, reward[controlled_agent_index])
            reward_shaped = reward

        return reward_shaped[controlled_agent_index]

    # def _get_reward_shaped(self, reward, done, controlled_agent_index):
    # 	if not done:
    # 		reward_shaped = [-1., -1.]
    # 	else:
    # 		if reward[0] != reward[1]:
    # 			reward_shaped = [reward[0] - 100, reward[1]] if reward[0] < reward[1] else [reward[0], reward[1] - 100]
    # 		else:
    # 			reward_shaped = [-1., -1.]
    # 	return reward_shaped[controlled_agent_index]

    def render(self, mode='human'):
        self.env.env_core.render()

    def close(self):
        pass

    def _transform_observation(self, frames):
        assert len(frames) == self.frame_stack
        obs = np.concatenate(list(frames), axis=0)
        return obs

    def wrestling_reward(self, obs):
        viewd_obs = obs[19:][:, 10:31]
        reward = len(viewd_obs[viewd_obs == 4])
        # max : 0.84
        return reward / 100

    def football_tablehockey_reward(self, obs, action):
        goal_viewed_obs = obs[20:29][:, 10:30]
        goal_reward = len(goal_viewed_obs[goal_viewed_obs == 2]) - 51  # -51 ~ 0
        goal_reward = goal_reward / 1000  # -0.051 ~ 0.00
        # # action_reward = action / 100
        # #
        # # if goal_reward > 1:
        # # 	goal_reward = goal_reward
        # # 	reward = goal_reward + action_reward
        # # else:
        # # 	reward = goal_reward
        #
        # if goal_reward > 30:
        # 	line_viewed_obs = obs[15:29][:, 10:30]
        # 	line_reward = len(line_viewed_obs[line_viewed_obs == 7]) * 10
        # else:
        # 	line_reward = 0
        #
        # reward = (goal_reward + line_reward) / 100
        ball_position = np.where(obs == 2)
        goal_line_position = np.where(obs == 7)

        ball_average_x = np.average(np.unique(ball_position[0]))
        ball_average_y = np.average(np.unique(ball_position[1]))
        ball_average_position = np.asarray([ball_average_x, ball_average_y])

        goal_line_average_x = np.average(np.unique(goal_line_position[0]))
        goal_line_average_y = np.average(np.unique(goal_line_position[1]))
        goal_line_average_position = np.asarray([goal_line_average_x, goal_line_average_y])

        if len(goal_viewed_obs[goal_viewed_obs == 2]) > 10:
            if np.nan_to_num(goal_line_average_x) and np.nan_to_num(ball_average_x):
                ball_goal_line_dist = np.linalg.norm(ball_average_position - goal_line_average_position)

                dist_reward = 50 - ball_goal_line_dist  # 0 ~ 50
                dist_reward = dist_reward / 25  # 0 ~ 2

                reward = goal_reward + dist_reward
            else:
                reward = goal_reward
        else:
            reward = -0.1

        return reward

    def running_reward(self, obs, energy):
        viewed_obs = obs[-20:-8][:, 10:31]
        mask = np.zeros_like(viewed_obs, dtype=bool)
        mask[viewed_obs[:][:] == 4] = True
        list_for_flag = [any(i) for i in mask]

        viewed_obs_forward = obs[-18:-12][:, 10:31]
        mask_forward = np.zeros_like(viewed_obs_forward, dtype=bool)
        mask_forward[viewed_obs_forward[:][:] == 4] = True
        list_for_forward_flag = [any(i) for i in mask_forward]

        penalty_wall = len(viewed_obs_forward[viewed_obs_forward == 6]) / 10
        penalty_step = -0.1

        if self.running_step_for_reward > 10:
            self.running_arrow = False
            self.running_step_for_reward = 0
        else:
            if self.running_arrow:
                self.running_step_for_reward += 1
                if sum(list_for_forward_flag) == 0:
                    self.running_arrow = False
                    self.running_step_for_reward = 0
                    return 1. - penalty_wall + penalty_step
            else:
                if sum(list_for_flag) > 5:
                    self.running_arrow = True
        return -penalty_wall + penalty_step

    def convert_obs_opponent_to_controlled(self, obs):
        obs = obs.copy()
        mask_eight = np.zeros_like(obs, dtype=bool)
        mask_eight[obs[:][:] == 8] = True

        mask_ten = np.zeros_like(obs, dtype=bool)
        mask_ten[obs[:][:] == 10] = True

        obs[mask_eight] += 2
        obs[mask_ten] -= 2

        return obs

    def linear_schedule(self, schdl, step):
        """
        Outputs values following a linear decay schedule.
        Adapted from https://github.com/facebookresearch/drqv2
        """
        try:
            return float(schdl)
        except ValueError:
            match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
            if match:
                init, final, duration = [float(g) for g in match.groups()]
                mix = np.clip(step / duration, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final
        raise NotImplementedError(schdl)

    def get_action_opponent(self, obs):
        coin = np.random.random()  # 0.0과 1.0사이의 임의의 값을 반환
        opponent_random_action_ratio = self.linear_schedule(self.args.opponent_agent_random_action_ratio,
                                                            self.total_steps)
        if coin > opponent_random_action_ratio and self.smart_competition:
            actions = self._get_action(obs)
            opponent_scaled_actions = self.get_scaled_action(actions)
        else:
            force = random.uniform(-100, 200)
            angle = random.uniform(-30, 30)
            opponent_scaled_actions = np.asarray([force, angle])

        if self.sub_game == "olympics-wrestling":
            return opponent_scaled_actions * opponent_random_action_ratio
        else:
            return opponent_scaled_actions

    def get_scaled_action(self, action):
        clipped_action = np.clip(action, -1.0, 1.0)

        scaled_action_0 = -100 + (clipped_action[0] + 1) / 2 * (200 - (-100))
        scaled_action_1 = -30 + (clipped_action[1] + 1) / 2 * (30 - (-30))

        return numpy.asarray([scaled_action_0, scaled_action_1])

    def _get_action(self, state: np.ndarray) -> float:
        state = np.array([state])
        state = torch.FloatTensor(state).to(self.device)

        action, dist = self.agent(state)

        return list(action.detach().cpu().numpy()).pop()

