import gym
import torch
from torch.backends import cudnn

import hkenv
import models
import trainer
import buffer

DEVICE = 'cuda'
cudnn.benchmark = True
test_path_list = ['saved/1703164894SW/besttrainonline.pt',
                  'saved/1702819178SW/besttrainonline.pt',
                  'saved/1703164894SW/bestonline.pt',
                  'saved/1702819178SW/bestonline.pt']


def get_model(env: gym.Env, n_frames: int, file_path=''):
    c, *shape = env.observation_space.shape
    m = models.SimpleExtractor(shape, n_frames * c)
    m = models.DuelingMLP(m, env.action_space.n, noisy=True)

    m = m.to(DEVICE)

    # modify below path to the weight file you have
    # tl = torch.load('saved/1673754862HornetPER/bestmodel.pt')
    # tl = torch.load('saved/1702297388HornetV2/bestonline.pt')
    # tl = torch.load('saved/1702722179Hornet/besttrainonline.pt')
    tl = torch.load(file_path)

    missing, unexpected = m.load_state_dict(tl, strict=False)
    if len(missing):
        print('miss:', missing)
    if len(unexpected):
        print('unexpected:', unexpected)
    return m


def main(p):
    n = 100  # test times
    n_frames = 4
    env = hkenv.HKEnv((160, 160), rgb=False, gap=0.165, w1=1, w2=1, w3=0)
    # env = hkenv.HKEnv((192, 192), rgb=False, gap=0.165, w1=1, w2=1, w3=0)
    # env = hkenv.HKEnvV2((192, 192), rgb=False, gap=0.17, w1=0.8, w2=0.5, w3=-8e-5)
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
