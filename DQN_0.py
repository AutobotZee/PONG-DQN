import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make('Pong-ram-v4').unwrapped

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
###############################################################################

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity # 余り
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

###############################################################################
###############################################################################

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
    
def get_screen():
    screen = env.render(mode='rgb_array').transpose((2,0,1))
    screen = screen[:, 34:194, :]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255# change shape
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)

env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),interpolation='none')
plt.title('Example extracted screen')
plt.show()
###############################################################################
###############################################################################
###############################################################################
###############################################################################

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

A=policy_net.state_dict()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()# random number between 0 and
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)# view changes the shape of a tensor
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations=[]


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)# mean(1) is the mean over column [N,M]->[N,1] 
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
###############################################################################
###############################################################################
###############################################################################
###############################################################################


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)# transitions = [Tr Tr Tr ...]
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])# Removing None
    state_batch = torch.cat(batch.state)# state_batch.shape = torch.Size([128, 3, 40, 90])
    action_batch = torch.cat(batch.action)# action_batch.shape = torch.Size([128, 1])
    reward_batch = torch.cat(batch.reward)# reward_batch.shape = torch.Size([128])
    state_action_values = policy_net(state_batch).gather(1, action_batch) 
  
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

###############################################################################
###############################################################################
###############################################################################
###############################################################################

num_episodes = 2
for i_episode in range(num_episodes):
    print(i_episode)
    env.reset()
    last_screen = get_screen()# tensor as a state
    current_screen = get_screen()
    state = current_screen - last_screen # the initial speed is zero
    for t in count():
        print(t)

        plt.figure()
        plt.imshow(current_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),interpolation='none')
        plt.show()

        action = select_action(state)# action = tensor([[1]]) or tensor([[0]])
        _, reward, done, _ = env.step(action.item())# action.item()= 1 or 0
        reward = torch.tensor([reward], device=device)# converting
        last_screen = current_screen
        current_screen = get_screen()# get pixel with the new state
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
# =============================================================================
# env.render()
# env.close()
# plt.ioff()
# plt.show()
# 
# =============================================================================

# =============================================================================
# plt.ion()
# s = env.reset()
# last_screen = current_screen
# current_screen = get_screen()# get pixel with the new state
# for _ in range(100):
#     state=current_screen-last_screen
#     action = policy_net(state).max(1)[1].view(1, 1)
#     s, _, done, _ = env.step(action.item())
#     last_screen = current_screen # tensor as a state
#     current_screen = get_screen()
# 
#     if done: break
# env.render()
# plt.ioff()
# plt.show()
# 
# =============================================================================


