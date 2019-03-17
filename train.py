from envs.make_env import make_env
import argparse, datetime
from tensorboardX import SummaryWriter
import numpy as np

from agents import Predators, Preyer
from normalized_env import NormalizedEnv
from utils import split_obs, merge_action
import time

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', default=100000, type=int)
parser.add_argument('--episode_length', default=500, type=int)
parser.add_argument('--memory_length', default=6000000, type=int)
parser.add_argument('--warmup', default=500, type=int)
parser.add_argument('--tau', default=0.001, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--use_cuda', default=True, type=bool)
parser.add_argument('--a_lr', default=0.0001, type=float)
parser.add_argument('--c_lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--render', default=False, type=bool)
parser.add_argument('--ou_theta', default=0.15, type=float)
parser.add_argument('--ou_mu', default=0.0, type=float)
parser.add_argument('--ou_sigma', default=0.2, type=float)
parser.add_argument('--epsilon_decay', default=50000, type=int)
parser.add_argument('--reward_coef', default=0.01, type=float)
parser.add_argument('--tensorboard', default=False, type=bool)
parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
args = parser.parse_args()

def main():

    env = make_env('simple_tag')
    env = NormalizedEnv(env)

    kwargs = dict()
    kwargs['config'] = args

    predator_model = Predators(16, 5, num_agent=3, **kwargs)
    preyer_model = Preyer(14, 5, **kwargs)
    if args.tensorboard:
        writer = SummaryWriter(log_dir='runs/'+args.log_dir)
    episode = 0

    while episode < args.max_episodes:

        state = env.reset()
        episode += 1
        step = 0
        predator_accum_reward = 0
        preyer_accum_reward = 0

        while True:
            state_predator, state_prayer = split_obs(state)

            if episode > args.warmup:
                action_predator = predator_model.choose_action(state_predator)
                action_prayer = preyer_model.choose_action(state_prayer)
            else:
                action_predator = predator_model.random_action()
                action_prayer = preyer_model.random_action()

            action = merge_action(action_predator, action_prayer)

            next_state, reward, done, info = env.step(action)
            step += 1

            predator_accum_reward = np.mean(reward[:3])
            preyer_accum_reward = reward[3]

            if step > args.episode_length:
                done = [True, True, True, True]

            if args.render and (episode % 10 == 1):
                env.render(mode='rgb_array')

            predator_model.memory(state[:3], action[:3], reward[:3], next_state[:3], done[:3])
            preyer_model.memory(state[3], action[3], reward[3], next_state[3], done[3])

            if episode > args.warmup:
                predator_model.train()
                preyer_model.train()

            if True in done:
                predator_c_loss, predator_a_loss = predator_model.getLoss()
                preyer_c_loss, preyer_a_loss = preyer_model.getLoss()
                print("[Episode %05d] reward_predator %3.1f reward_preyer %3.1f predator_c_loss %3.1f predator_a_loss %3.1f preyer_c_loss %3.1f preyer_a_loss %3.1f" % \
                      (episode, predator_accum_reward, preyer_accum_reward, predator_c_loss, predator_a_loss, preyer_c_loss, preyer_a_loss))
                if args.tensorboard:
                    writer.add_scalar(tag='debug/memory_length', global_step=episode, scalar_value=len(predator_model.replay_buffer))
                    writer.add_scalar(tag='debug/predator_epsilon', global_step=episode, scalar_value=predator_model.epsilon)
                    writer.add_scalar(tag='debug/preyer_epsilon', global_step=episode, scalar_value=preyer_model.epsilon)
                    writer.add_scalar(tag='perf/reward_predator', global_step=episode, scalar_value=predator_accum_reward)
                    writer.add_scalar(tag='perf/reward_preyer', global_step=episode, scalar_value=preyer_accum_reward)
                    writer.add_scalar(tag='loss/predator_c_loss', global_step=episode, scalar_value=predator_c_loss)
                    writer.add_scalar(tag='loss/predator_a_loss', global_step=episode, scalar_value=predator_a_loss)
                    writer.add_scalar(tag='loss/preyer_c_loss', global_step=episode, scalar_value=preyer_c_loss)
                    writer.add_scalar(tag='loss/preyer_a_loss', global_step=episode, scalar_value=preyer_a_loss)

                predator_model.reset()
                preyer_model.reset()
                break

            state = next_state
    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()



