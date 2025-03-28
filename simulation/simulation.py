import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization
import gymnasium as gym
from gymnasium import spaces
import stable_baselines3 as sb3
import matplotlib.cm as cm  # Import colormaps

class CuttingEnv(gym.Env):
    """
    Custom RL environment for optimizing cutting strategies dynamically
    """
    def __init__(self, initial_params=None):
        super(CuttingEnv, self).__init__()
        
        self.action_space = spaces.Box(
            low=np.array([0, 0.5, 1, -0.02], dtype=np.float32),
            high=np.array([45, 5, 5, -0.001], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        self.vegetable = 'lettuce'
        self.current_step = 0
        self.initial_params = np.array(list(initial_params.values()) if initial_params else [10, 1.5, 2, -0.005], dtype=np.float32)
        self.episode_rewards = []  #store rewards 
        self.all_episode_rewards = [] #rewards per episopde
        self.episode_actions = [] #actions per episode 


    def step(self, action):
        cutting_angle, cutting_force, cutting_rhythm, cutting_speed = action
        results = simulate_cutting(self.vegetable, cutting_angle, cutting_force, cutting_rhythm, cutting_speed)

        if not results:
            #penalty
            reward = -1.0
            displacement = 0.0
        else:
            displacement = np.mean([
                np.abs(data['cut_position'][2] - data['initial_position'][2])
                for data in results.values()
            ])
            reward = - (displacement + cutting_force * 0.1)

            if np.isnan(displacement):
                reward = -1.0

        self.episode_rewards.append(reward)
        self.episode_actions.append(action)

        self.current_step += 1
        done = self.current_step >= 50
        truncated = False

        obs = np.array([cutting_angle, cutting_force, cutting_rhythm, cutting_speed], dtype=np.float32)
        
        if np.any(np.isnan(obs)):
            obs = np.nan_to_num(obs, nan=0.0)

        if done:
            total_episode_reward = sum(self.episode_rewards)
            self.all_episode_rewards.append(total_episode_reward)

        return obs, reward, done, truncated, {}


    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.episode_rewards = []  #reset reward for current episode
        self.episode_actions = [] #also actions
        return self.initial_params.copy(), {}

    def render(self, mode='human'):
        pass

def simulate_cutting(vegetable, cutting_angle=0, cutting_force=1, cutting_rhythm=1, cutting_speed=-0.005, cutting_threshold=0.12, num_slices=6):
    """
    Simulates a vegetable cutting stroke
    """
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0, 0, -0.05], p.getQuaternionFromEuler([0, 0, 0]))
    
    vegetable_properties = {
        'cucumber': {'height': 0.12, 'radius': 0.02, 'mass': 0.1},
        'potato': {'height': 0.08, 'radius': 0.03, 'mass': 0.2},
        'lettuce': {'height': 0.15, 'radius': 0.05, 'mass': 0.05}
    }
    
    if vegetable not in vegetable_properties:
        raise ValueError("Invalid vegetable selection")
    
    properties = vegetable_properties[vegetable]
    slice_thickness = properties['height'] / num_slices
    
    vegetable_parts = []
    initial_positions = {}
    
    for i in range(num_slices):
        slice_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=properties['radius'], height=slice_thickness)
        body_id = p.createMultiBody(baseMass=properties['mass'], baseCollisionShapeIndex=slice_id,
                                    basePosition=[0, 0, 0.1 + i * slice_thickness], baseOrientation=[0, 0, 0, 1])
        vegetable_parts.append(body_id)
        initial_positions[body_id] = p.getBasePositionAndOrientation(body_id)[0]
    
    knife_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.005, 0.02])
    knife_id = p.createMultiBody(baseMass=cutting_force, baseCollisionShapeIndex=knife_col,
                                    basePosition=[0, 0, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, np.radians(cutting_angle)]))
    
    step_count = 0
    collision_data = {}
    while True:
        step_count += 1
        knife_pos, _ = p.getBasePositionAndOrientation(knife_id)
        new_knife_pos = [knife_pos[0], knife_pos[1], knife_pos[2] + cutting_speed]
        if new_knife_pos[2] <= cutting_threshold:
            break
        p.resetBasePositionAndOrientation(knife_id, new_knife_pos, p.getQuaternionFromEuler([0, 0, np.radians(cutting_angle)]))
        for part in vegetable_parts:
            contact_points = p.getContactPoints(knife_id, part)
            if contact_points:
                part_pos, part_ori = p.getBasePositionAndOrientation(part)
                new_part_pos = [part_pos[0] + np.random.uniform(-0.005, 0.005),
                                part_pos[1] + np.random.uniform(-0.005, 0.005),
                                part_pos[2] + np.random.uniform(0.005, 0.01)]
                p.resetBasePositionAndOrientation(part, new_part_pos, part_ori)
                if part not in collision_data:
                    collision_data[part] = {
                        "initial_position": initial_positions[part],
                        "cut_position": new_part_pos,
                        "step_cut": step_count,
                        "cutting_angle": cutting_angle,
                        "cutting_force": cutting_force,
                        "cutting_rhythm": cutting_rhythm
                    }
        p.stepSimulation()
    p.disconnect()
    return collision_data

def optimize_cutting():
    def objective(cutting_angle, cutting_force, cutting_rhythm, cutting_speed):
        results = simulate_cutting('lettuce', cutting_angle, cutting_force, cutting_rhythm, cutting_speed)
        
        if not results:
            return -1.0  #failed cut penalty
        
        displacement = np.mean([
            np.abs(data['cut_position'][2] - data['initial_position'][2])
            for data in results.values()
        ])
        return -displacement

    
    optimizer = BayesianOptimization(
        f=objective,
        pbounds={'cutting_angle': (0, 45), 'cutting_force': (0.5, 5), 'cutting_rhythm': (1, 5), 'cutting_speed': (-0.02, -0.001)},
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=15)

    #plot
    plot_bayesian_optimization(optimizer)

    return optimizer.max['params']

def plot_bayesian_optimization(optimizer):
    """Plots the results of the Bayesian Optimization."""
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    y_vals = optimizer.res
    best_y = [y_vals[0]['target']]
    for y in y_vals[1:]:
        if y['target'] > best_y[-1]:
             best_y.append(y['target'])
        else:
             best_y.append(best_y[-1])
    plt.plot(best_y)
    plt.xlabel('Iteration')
    plt.ylabel('Best Reward')
    plt.title('Bayesian Optimization Convergence')

     #heatmap
    plt.subplot(1, 2, 2)
    param_values = [res["params"] for res in optimizer.res]
    param_names = list(optimizer.space.keys)
    param_matrix = np.array([[p[name] for name in param_names] for p in param_values])
    plt.imshow(param_matrix, aspect='auto', cmap=cm.viridis)
    plt.yticks(range(len(optimizer.res)), [f"Iter {i}" for i in range(len(optimizer.res))])
    plt.xticks(range(len(param_names)), param_names, rotation=45)
    plt.colorbar(label='Parameter Value')
    plt.title('Parameter Values during Optimization')
    plt.tight_layout()
    plt.show()
    


def train_rl_agent(initial_params):
    env = CuttingEnv(initial_params)
    model = sb3.PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)

#plot rl
    plot_rl_training(env)
    return model

def plot_rl_training(env):
    """Plots the rewards over time during RL training"""
    if not env.all_episode_rewards: 
        print("No rewards to plot yet.")
        return
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.plot(env.all_episode_rewards) 
    plt.xlabel('Episode') 
    plt.ylabel('Total Reward') 
    plt.title('RL Training Rewards per Episode')

    if env.episode_actions:
        plt.subplot(1,2,2)
        actions = np.array(env.episode_actions)
        param_names = ['cutting_angle', 'cutting_force', 'cutting_rhythm', 'cutting_speed']
        for i in range(actions.shape[1]):
            plt.plot(actions[:,i], label = param_names[i])
        plt.xlabel("Step")
        plt.ylabel("Action Value")
        plt.legend()
        plt.title("Actions over time")


    plt.tight_layout()
    plt.show()



optimized_params = optimize_cutting()
print("Optimized Parameters from Bayesian Optimization:", optimized_params)
trained_agent = train_rl_agent(optimized_params)