import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import os
import multiprocessing

    # Define the worker function for each process
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError
            


# Wrapper to make the environment function picklable
class EnvFnWrapper(object):
    def __init__(self, env_fn):
        self.env_fn = env_fn
    def x(self):
        return self.env_fn()
    

# ParallelEnv class to manage multiple environment processes
class ParallelEnv:
    def __init__(self, num_envs, env_fn):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.waiting = False
        self.closed = False
        self.num_envs = num_envs

        self.remotes, self.work_remotes = zip(*[multiprocessing.Pipe() for _ in range(num_envs)])
        self.processes = []

        for work_remote, remote in zip(self.work_remotes, self.remotes):
            env_fn_wrapper = EnvFnWrapper(env_fn)
            process = multiprocessing.Process(target=worker, args=(work_remote, remote, env_fn_wrapper))
            process.daemon = True
            process.start()
            work_remote.close()

        self.remotes = self.remotes
    
    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, dones, infos = zip(*results)
        # Convert observations, rewards, and dones to PyTorch tensors
        # print(obs)
        obs = torch.stack(obs, dim=0).to(self.device)  # Move directly to the desired device

        # obs = torch.stack([torch.tensor(o, device=self.device) for o in obs])
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        return obs, rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        # print(type(self.remotes[0]))
        return torch.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

