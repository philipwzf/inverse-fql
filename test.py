import pickle
import dsrl
import gymnasium
import safety_gymnasium
import numpy as np
from PIL import Image

# env = gymnasium.make('OfflinePointCircle1Gymnasium-v0', render_mode='rgb_array_list')
env = safety_gymnasium.make('SafetyPointCircle1-v0',render_mode='rgb_array',camera_id=1)
obs = env.reset()
frame = np.array(env.render())
print(frame.shape)

img = Image.fromarray(frame, 'RGB')

img = img.save("camera=1.png")

print(env.sim.model.camera_names)
asd
dataset = env.get_dataset()
terminals = dataset['terminals']
observations = dataset['observations']
print(observations.shape)
print(observations[0])


terminals = np.zeros_like(dataset['rewards'])
masks = np.zeros_like(dataset['rewards'])  # Indicate whether we should bootstrap from the next state.
rewards = dataset['rewards'].copy().astype(np.float32)
costs = dataset['costs'].copy().astype(np.float32)

for i in range(len(terminals) - 1):
    if (
        np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
        or dataset['terminals'][i] == 1.0
    ):
        terminals[i] = 1
    else:
        terminals[i] = 0
    masks[i] = 1 - dataset['terminals'][i]
masks[-1] = 1 - dataset['terminals'][-1]
terminals[-1] = 1
print(terminals.sum())
print(rewards.sum())
print(costs.sum()/terminals.sum())

# Replace 'your_file.pkl' with the path to your pickle file.
file = "/home/philipwang/data/fql/exp/fql/Debug/sd000_20250209_154632/params_1000000.pkl"
with open(file, 'rb') as f:
    data = pickle.load(f)

# Now 'data' holds the unpickled object.
print(data['agent']['network']['params'].keys())