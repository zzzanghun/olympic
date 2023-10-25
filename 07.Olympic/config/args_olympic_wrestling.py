import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
from e_utils import boolean_argument


def get_args(rest_args):
    """PPOAgent.
        Parameters:
            device: cpu or gpu acelator.
            make_env: factory that produce environment.
            continuous: True of environments with continuous action space.
            obs_dim: dimension od observaion.
            act_dim: dimension of action.
            gamma: coef for discount factor.
            lamda: coef for general adversial estimator (GAE).
            entropy_coef: coef of weighting entropy in objective loss.
            epsilon: clipping range for actor objective loss.
            actor_lr: learnig rate for actor optimizer.
            critic_lr: learnig rate for critic optimizer.
            value_range: clipping range for critic objective loss.
            rollout_len: num t-steps per one rollout.
            total_rollouts: num rollouts.
            num_epochs: num weights updation iteration for one policy update.
            batch_size: data batch size for weights updating
            actor: model for predction action.
            critic: model for prediction state values.
            plot_interval: interval for plotting train history.
            solved_reward: desired reward.
            plot_interval: plot history log every plot_interval rollouts.
            path2save_train_history: path to save training history logs.
            """
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='olympics-wrestling', help='name of environment')

    # device
    parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu acelator')

    # coefficients
    parser.add_argument('--gamma', type=float, default=0.9, help='coef for discount factor')
    parser.add_argument('--lamda', type=float, default=0.95, help='coef for general adversial estimator (GAE)')
    parser.add_argument('--entropy_coef', type=float, default=0.005, help='coef for general adversial estimator (GAE)')
    parser.add_argument('--epsilon', type=float, default=0.2, help='clipping range for actor objective loss')
    parser.add_argument('--value_range', type=float, default=0.5, help='clipping range for critic objective loss')

    # other hyperparameters
    parser.add_argument('--rollout_len', type=int, default=4000, help='num t-steps per one rollout')
    parser.add_argument('--total_rollouts', type=int, default=1000, help='num rollouts')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='num weights updation iteration for one policy update')
    parser.add_argument('--batch_size', type=int, default=128, help='data batch size for weights updating')

    # agent net
    parser.add_argument('--obs_dim', type=tuple, default=(4, 40, 40), help='dimension od observaion')
    parser.add_argument('--continuous', type=boolean_argument, default=True,
                        help='True of environments with continuous action space')
    parser.add_argument('--act_dim', type=int, default=2, help='dimension of action')

    # agent nets optimizers
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='learning rate for actor optimizer')
    parser.add_argument('--critic_lr', type=float, default=5e-4, help='learning rate for actor optimizer')

    # etc.
    parser.add_argument('--is_evaluate', type=boolean_argument, default=False, help='for evaluation')
    parser.add_argument('--solved_reward', type=int, default=0, help='desired reward')
    parser.add_argument('--plot_interval', type=int, default=1, help='interval for plotting train history')
    parser.add_argument('--print_episode_interval', type=int, default=10, help='interval for printing train history')

    # olympic.
    parser.add_argument('--render_over_train', type=boolean_argument, default=True, help='render over train')
    parser.add_argument('--controlled_agent_index', type=int, default=1, help='controlled agent index')
    parser.add_argument('--opponent_agent_random_action_ratio', type=str, default='linear(0.5, 0.1, 10000)',
                        help='random action annealing')
    parser.add_argument('--frame_stack', type=int, default=4, help='frame stack')
    parser.add_argument('--frame_skipping', type=int, default=4, help='frame skipping')
    parser.add_argument('--wandb_use', type=boolean_argument, default=False, help='wandb_use')
    parser.add_argument('--target_update_interval', type=int, default=10, help='actor target update interval for smart competition')
    parser.add_argument('--smart_competition', type=boolean_argument, default=False, help='smart_competition')
    parser.add_argument('--load_model', type=boolean_argument, default=False, help='load previous model')
    parser.add_argument('--load_model_time', type=str, default="10_3_9", help='load previous model time')


    return parser.parse_args(rest_args)
