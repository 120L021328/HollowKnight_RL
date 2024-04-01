import gym
import torch
from torch.backends import cudnn

import hkenv
import models
import trainer
import buffer

DEVICE = 'cuda'
cudnn.benchmark = True
# modify below path to the weight file you have
test_path_list = ['saved/1702905513Hornet/bestonline.pt',
                  'saved/1702722179Hornet/bestonline.pt'
                  ]


def get_model(env: gym.Env, n_frames: int, file_path=''):
    c, *shape = env.observation_space.shape
    m = models.SimpleExtractor(shape, n_frames * c)
    m = models.DuelingMLP(m, env.action_space.n, noisy=True)
    m = m.to(DEVICE)
    m.load_state_dict(torch.load(file_path))
    return m


def main(p):
    n = 1  # test times
    n_frames = 4
    env = hkenv.HKEnv((160, 160), rgb=False, gap=0.165, w1=1, w2=1, w3=0)
    m = get_model(env, n_frames, p)
    replay_buffer = buffer.MultistepBuffer(100000, n=10, gamma=0.99)
    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.99, eps=0.,
                          eps_func=(lambda val, step: 0.),
                          target_steps=6000,
                          learn_freq=1,
                          model=m,
                          lr=9e-5,
                          lr_decay=False,
                          criterion=torch.nn.MSELoss(),
                          batch_size=32,
                          device=DEVICE,
                          is_double=True,
                          drq=True,
                          svea=False,
                          reset=0,
                          no_save=True)

    total_reward = 0
    total_win = 0
    for i in range(n):
        rew, w = dqn.evaluate()
        total_reward += rew
        if w:
            total_win += 1
        if i % 10 == 9:
            print('finished %d times' % (i+1))
    average_rew = total_reward / n
    print("rewards: %f, win times: %d" % (average_rew, total_win))


if __name__ == '__main__':
    for path in test_path_list:
        main(path)
