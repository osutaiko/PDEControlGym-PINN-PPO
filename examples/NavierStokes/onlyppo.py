import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
import time 
import os # Added for path checking
from tqdm import tqdm
# Assuming pde_control_gym is installed and NSReward is accessible
# If not, you might need to adjust the import path based on your project structure
# from pde_control_gym.src import NSReward 
# For now, as NSReward is not fully defined, I'll create a placeholder if not found
try:
    from pde_control_gym.src import NSReward
except ImportError:
    print("Warning: pde_control_gym.src.NSReward not found. Using a placeholder reward.")
    class NSReward: # Placeholder
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, state, action, previous_state, previous_action, U_ref, action_ref):
            return 0 # Dummy reward

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO

# THIS EXAMPLE SOLVES THE NavierStokes PROBLEM based on optimization

# Set initial condition function here
def getInitialCondition(X_grid_shape): # Modified to accept shape for clarity
    # The environment should pass the actual grid or its shape.
    # If X_grid_shape is just a scalar from the original X, this will be problematic.
    # Assuming X_grid_shape is something like (Ny, Nx) or similar for u,v,p fields.
    u = np.random.uniform(-5, 5, size=X_grid_shape)
    v = np.random.uniform(-5, 5, size=X_grid_shape)
    p = np.random.uniform(-5, 5, size=X_grid_shape)
    return u, v, p

# Set up boundary conditions here
boundary_condition = {
    "upper": ["Controllable", "Dirchilet"], 
    "lower": ["Dirchilet", "Dirchilet"], 
    "left": ["Dirchilet", "Dirchilet"], 
    "right": ["Dirchilet", "Dirchilet"], 
}

# Timestep and spatial step for PDE Solver
T_sim_duration = 0.2 # Renamed T to avoid conflict with matplotlib.pyplot.T
dt = 1e-3
dx, dy = 0.05, 0.05
X_domain, Y_domain = 1.0, 1.0 # Renamed X, Y to avoid conflict

# --- IMPORTANT: Ensure target.npz path is correct ---
target_file_path = 'C:\\Users\\USER\\Downloads\\PDEControlGym-main\\PDEControlGym-main\\examples\\NavierStokes\\target.npz'
if not os.path.exists(target_file_path):
    print(f"ERROR: Target file not found at {target_file_path}")
    print("Please ensure the path is correct or provide a valid target file.")
    # Create dummy target data if file not found, to allow script to run for structure demo
    print("Using dummy target data for demonstration purposes.")
    num_timesteps_target = int(T_sim_duration / dt)
    Nx_target = int(X_domain / dx)
    Ny_target = int(Y_domain / dy)
    u_target = np.zeros((num_timesteps_target, Ny_target, Nx_target)) # Typically (NT, Ny, Nx)
    v_target = np.zeros((num_timesteps_target, Ny_target, Nx_target))
else:
    u_target = np.load(target_file_path)['u']
    v_target = np.load(target_file_path)['v']

# Assuming u_target, v_target are (NT, Ny, Nx) or similar, need to match env's expectation
# The original code had np.stack([u_target, v_target], axis=-1)
# If u_target is (NT, Nx, Ny), then desire_states becomes (NT, Nx, Ny, 2)
# If u_target is (NT, Ny, Nx), then desire_states becomes (NT, Ny, Nx, 2)
# Let's assume Ny, Nx order for typical image/grid processing: (NT, Ny, Nx)
if u_target.ndim == 3 and v_target.ndim == 3:
    desire_states = np.stack([u_target, v_target], axis=-1)
else:
    print(f"Warning: Unexpected shape for u_target ({u_target.shape}) or v_target ({v_target.shape}).")
    print("Creating dummy desire_states.")
    dummy_Ny = int(Y_domain/dy)
    dummy_Nx = int(X_domain/dx)
    dummy_NT = int(T_sim_duration/dt)
    desire_states = np.zeros((dummy_NT, dummy_Ny, dummy_Nx, 2))


NS2DParameters = {
        "T": T_sim_duration, 
        "dt": dt, 
        "X": X_domain,
        "dx": dx, 
        "Y": Y_domain,
        "dy":dy,
        "action_dim": 1, 
        "reward_class": NSReward(0.1), # Make sure NSReward is correctly defined/imported
        "normalize": False, 
        # MODIFIED: Lambda now accepts one argument (ignored) to match environment's call signature
        "reset_init_condition_func": lambda _env_arg_ignored: getInitialCondition((int(Y_domain/dy), int(X_domain/dx))),
        "boundary_condition": boundary_condition,
        "U_ref": desire_states, 
        "action_ref": 2.0 * np.ones(1000), 
        # Add render_mode if you want to use env.render() directly during visualization
        # "render_mode": "human", 
}

# Make the NavierStokes PDE gym
# IMPORTANT: The environment "PDEControlGym-NavierStokes2D" must be registered with Gymnasium
try:
    env = gym.make("PDEControlGym-NavierStokes2D", **NS2DParameters)
except gym.error.NameNotFound as e:
    print(f"Error: Environment 'PDEControlGym-NavierStokes2D' not found.")
    print("Please ensure that the PDEControlGym is correctly installed and the environment is registered.")
    print("Exiting.")
    exit()


# Save a checkpoint every 1000 steps (was 10000, user code has 1000)
checkpoint_callback = CheckpointCallback(
  save_freq=1000, # User's code has 1000, not 10000
  save_path="./logsPPO_NS2D", # Changed path slightly to avoid conflict if other PPO logs exist
  name_prefix="rl_model_ns2d",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_NS2D/")
# Train for 2e5 timesteps (user's code)
total_training_timesteps = int(2e5)
model.learn(total_timesteps=total_training_timesteps, callback=checkpoint_callback)
model.save("./logsPPO_NS2D/rl_model_ns2d_final") # Save final model

print("Training finished.")

# === Visualization Part (Added after model.learn()) ===
print("Starting visualization...")

# --- Option 1: Use env.render() (if available and properly configured) ---
# To use this, the environment needs to support a render_mode like 'human' or 'rgb_array'
# and you might need to pass `render_mode` during `gym.make`.
# Example:
# try:
#     print("\nAttempting visualization with env.render()...")
#     # Create a new env for rendering or ensure the existing one supports it
#     # vis_env_params = NS2DParameters.copy()
#     # vis_env_params["render_mode"] = "human" 
#     # vis_env = gym.make("PDEControlGym-NavierStokes2D", **vis_env_params)
#     # model_to_visualize = PPO.load("./logsPPO_NS2D/rl_model_ns2d_final.zip", env=vis_env)
#     
#     # obs, info = vis_env.reset()
#     # for i in range(200): # Visualize for 200 steps
#     #     action, _ = model_to_visualize.predict(obs, deterministic=True)
#     #     obs, _, terminated, truncated, _ = vis_env.step(action)
#     #     frame = vis_env.render()
#     #     time.sleep(0.05) # Slow down for visibility
#     #     if terminated or truncated:
#     #         print(f"Episode finished at step {i+1}.")
#     #         obs, info = vis_env.reset()
#     # vis_env.close()
# except Exception as e:
#     print(f"env.render() based visualization failed: {e}")
#     print("Skipping env.render() visualization. Check if environment supports rendering and 'render_mode' is set.")

# --- Option 2: Custom plotting with matplotlib (More control, requires understanding obs structure) ---
print("\nAttempting custom plot of u, v, p fields using matplotlib...")

# Load the final model or a specific checkpoint
model_path_to_load = "./logsPPO_NS2D/rl_model_ns2d_final.zip" 
# Or a specific checkpoint:
# model_path_to_load = f"./logsPPO_NS2D/rl_model_ns2d_{total_training_timesteps}_steps.zip"

if not os.path.exists(model_path_to_load):
    print(f"Model file not found at {model_path_to_load}. Cannot perform custom plotting.")
else:
    # Create a new environment instance for plotting, or reuse 'env' if appropriate
    # Reusing 'env' here. If 'env' was a VecEnv for training, you'd need to handle it.
    # The provided code uses a single env, so direct reuse is fine.
    # However, it's safer to create a new instance for plotting to avoid state issues.
    try:
        plot_env = gym.make("PDEControlGym-NavierStokes2D", **NS2DParameters)
    except gym.error.NameNotFound as e:
        print(f"Error creating plot_env: Environment 'PDEControlGym-NavierStokes2D' not found.")
        print("Exiting visualization part.")
        plot_env = None # Ensure plot_env is defined for the finally block
        exit()
        
    if plot_env:
        model_to_plot = PPO.load(model_path_to_load, env=plot_env)

        obs, info = plot_env.reset()
        # Take one step with the policy to get a representative state for plotting
        action, _ = model_to_plot.predict(obs, deterministic=True)
        current_obs, reward, terminated, truncated, info = plot_env.step(action)

        # --- Critical part: Extracting u, v, p from 'current_obs' ---
        # This depends HEAVILY on how "PDEControlGym-NavierStokes2D" structures its observation.
        # You MUST verify this based on the environment's documentation or observation_space.
        
        print(f"Observation space shape: {plot_env.observation_space.shape}")
        # print(f"Sample observation (first 10 elements): {current_obs[:10]}")
        # print(f"Sample observation length: {len(current_obs)}")

        Nx = int(X_domain / dx)  # Number of grid points in x
        Ny = int(Y_domain / dy)  # Number of grid points in y

        u_field, v_field, p_field = None, None, None
        fields_extracted = False

        # Common ways observations are structured for grid-based environments:
        # 1. Flattened array: [u_flat, v_flat, p_flat] or [u_flat, v_flat]
        # 2. Dictionary: {'u': u_2d_array, 'v': v_2d_array, 'p': p_2d_array} (less common for SB3 default MlpPolicy)
        
        obs_shape = plot_env.observation_space.shape
        if obs_shape is not None and len(obs_shape) == 1: # Likely a flattened array
            flat_obs_len = obs_shape[0]
            
            if flat_obs_len == Nx * Ny * 3: # Assuming u, then v, then p, each (Ny, Nx) flattened
                print(f"Observation is flat array of length {flat_obs_len}. Assuming u, v, p concatenated (Ny,Nx order).")
                u_flat_end = Ny * Nx
                v_flat_end = 2 * Ny * Nx
                u_field = current_obs[0 : u_flat_end].reshape(Ny, Nx)
                v_field = current_obs[u_flat_end : v_flat_end].reshape(Ny, Nx)
                p_field = current_obs[v_flat_end : ].reshape(Ny, Nx)
                fields_extracted = True
            elif flat_obs_len == Nx * Ny * 2: # Assuming u, then v, p is not in observation
                print(f"Observation is flat array of length {flat_obs_len}. Assuming u, v concatenated (Ny,Nx order). P might be missing.")
                u_flat_end = Ny * Nx
                u_field = current_obs[0 : u_flat_end].reshape(Ny, Nx)
                v_field = current_obs[u_flat_end : ].reshape(Ny, Nx)
                p_field = np.zeros((Ny, Nx)) # Placeholder for p
                print("Warning: Pressure (p) field seems to be missing from observation, plotted as zeros.")
                fields_extracted = True
            # Add more checks if you know other possible structures (e.g., if data is (Nx, Ny) then flattened)
            elif flat_obs_len == 3 * Nx * Ny and (Nx != Ny): # Check if order might be (Nx,Ny) then flattened
                 print(f"Observation is flat array of length {flat_obs_len}. Trying to interpret as (u_NxNy, v_NxNy, p_NxNy).")
                 # This case requires careful reshaping if the environment flattens (Nx,Ny) arrays.
                 # For now, stick to (Ny,Nx) as it's more common for image-like data.
                 pass


        # If you have a MultiDiscrete or Dict observation space, the extraction will be different.
        # Example for Dict space (you'd need to check `isinstance(plot_env.observation_space, gym.spaces.Dict)`)
        # if isinstance(current_obs, dict) and 'u' in current_obs and 'v' in current_obs:
        #     u_field = current_obs['u'] # Assuming these are already 2D arrays (Ny, Nx)
        #     v_field = current_obs['v']
        #     p_field = current_obs.get('p', np.zeros((Ny, Nx))) # Get p if available
        #     fields_extracted = True
        #     if not (u_field.shape == (Ny, Nx) and v_field.shape == (Ny, Nx)):
        #          print(f"Warning: u/v fields from Dict obs have unexpected shapes: u:{u_field.shape}, v:{v_field.shape}")
        #          fields_extracted = False


        if not fields_extracted:
            print("\nCould not automatically extract u,v,p fields from the observation.")
            print("Please inspect 'plot_env.observation_space' and the structure of 'current_obs'.")
            print("You may need to adjust the reshaping logic above based on your environment's specifics.")
        else:
            # Create grid for plotting (cell centers or edges, depending on what data represents)
            # Using cell centers for contourf often looks good.
            x_coords = np.linspace(dx/2, X_domain - dx/2, Nx)
            y_coords = np.linspace(dy/2, Y_domain - dy/2, Ny)
            X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

            plt.figure(figsize=(18, 5)) # Adjusted figure size

            plt.subplot(1, 3, 1)
            if u_field is not None:
                # contourf expects X, Y, Z where Z is (Y.shape[0], X.shape[0])
                # If u_field is (Ny, Nx), and X_grid, Y_grid are from meshgrid(x_coords, y_coords)
                # then X_grid is (Ny, Nx) and Y_grid is (Ny, Nx)
                cf_u = plt.contourf(X_grid, Y_grid, u_field, levels=50, cmap='viridis')
                plt.colorbar(cf_u)
                plt.title("U Velocity")
            else:
                plt.title("U Velocity (Not Available)")
            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")
            plt.axis('scaled') # Ensures aspect ratio is 1:1 for spatial plots

            plt.subplot(1, 3, 2)
            if v_field is not None:
                cf_v = plt.contourf(X_grid, Y_grid, v_field, levels=50, cmap='viridis')
                plt.colorbar(cf_v)
                plt.title("V Velocity")
            else:
                plt.title("V Velocity (Not Available)")
            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")
            plt.axis('scaled')

            plt.subplot(1, 3, 3)
            if p_field is not None:
                if np.any(p_field) or p_field.shape == (0,0): # Check if p_field is not just zeros or empty
                     cf_p = plt.contourf(X_grid, Y_grid, p_field, levels=50, cmap='viridis')
                     plt.colorbar(cf_p)
                     plt.title("Pressure P")
                else:
                     plt.title("Pressure P (Zeros or Placeholder)")
                     plt.text(0.5, 0.5, 'P field is zeros\nor placeholder', 
                              horizontalalignment='center', verticalalignment='center', 
                              transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))
            else:
                plt.title("Pressure P (Not Available)")

            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")
            plt.axis('scaled')
            
            plt.suptitle("Navier-Stokes Flow Fields (after training)", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
            plt.show()

        # Important: Close the environment if you are done with it
        if plot_env:
            plot_env.close()
        # Close the original training environment if it's different and still open
        # In this script, 'env' is used for training, and 'plot_env' might be the same instance or a new one.
        # If 'plot_env' was a new instance, 'env' (training env) should also be closed if no longer needed.
        # However, 'env' is implicitly closed when the script ends if not explicitly closed.
        # For safety, if they are different objects:
        # if 'env' in locals() and id(env) != id(plot_env) and hasattr(env, 'close'):
        #     env.close()


print("Script finished.")
