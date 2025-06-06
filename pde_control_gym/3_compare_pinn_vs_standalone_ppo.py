import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import pandas as pd

# --- 기본 환경 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- 시각화/평가 장치: {DEVICE} ---")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- 경로 설정 ---
PRETRAINED_PINN_MODEL_PATH = os.path.join(ROOT_DIR, "time_control_pinn.pth")
PINN_PPO_POLICY_FILE_PATH = os.path.join(ROOT_DIR, "ppo_pinn_control_policy.zip")
STANDALONE_PPO_POLICY_FILE_PATH = os.path.join(ROOT_DIR, "ppo_standalone_policy.zip")

EVALUATION_RESULTS_SAVE_PATH = "./final_comparison_log/ppo_scenario_comparison.csv"
PLOT_SAVE_DIR = "./final_comparison_log/"
if not os.path.exists(PLOT_SAVE_DIR):
    os.makedirs(PLOT_SAVE_DIR)
SAVE_PLOTS = True

# --- 물리 상수 (PINN 환경용) ---
D_CHANNEL_FOR_PINN_ENV = 1.0
PINN_ENV_Y_MIN_CFG = 0.0
PINN_ENV_Y_MAX_CFG = D_CHANNEL_FOR_PINN_ENV

FIELD_COLOR_LEGENDS = {
    "u-velocity": [0, 1.5],
    "v-velocity": [-0.3, 0.3],
    "pressure (p)": None
}

# --- PINN 모델 정의 (PINN-PPO용) ---
class TimeControlPINN_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_layers = 5
        neurons_per_layer = 128
        layers = [input_dim] + [neurons_per_layer] * hidden_layers + [3]
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )
        self.activation = torch.tanh
    def forward(self, xytc):
        for i, linear_layer in enumerate(self.linears[:-1]):
            xytc = self.activation(linear_layer(xytc))
        return self.linears[-1](xytc)

# --- PINN 기반 Gym 환경 정의 (PINN-PPO용) ---
class PINNNavierStokesEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 30}
    def __init__(self, pretrained_pinn_model_path, env_config):
        super().__init__()
        self.T_total = env_config["T_sim"]
        self.dt = env_config["dt_sim"]
        self.num_control_params = env_config["num_control_params"]
        self.current_step_env = 0
        self.max_steps_env = int(self.T_total / self.dt)

        pinn_input_dim = 2 + 1 + self.num_control_params
        self.pinn_model = TimeControlPINN_Model(input_dim=pinn_input_dim).to(DEVICE)
        try:
            self.pinn_model.load_state_dict(torch.load(pretrained_pinn_model_path, map_location=DEVICE))
        except FileNotFoundError:
            raise FileNotFoundError(f"PINN 모델 파일 없음: {pretrained_pinn_model_path}")
        except Exception as e:
            raise RuntimeError(f"PINN 모델 로드 오류 ({pretrained_pinn_model_path}): {e}")
        self.pinn_model.eval()

        self.action_space = spaces.Box(
            low=env_config.get("action_low", -1.0),
            high=env_config.get("action_high", 1.0),
            shape=(self.num_control_params,), dtype=np.float32
        )
        self.nx_obs = env_config.get("nx_obs", 20)
        self.ny_obs = env_config.get("ny_obs", 20)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nx_obs, self.ny_obs, 3), dtype=np.float32)

        self.domain_L_pinn = env_config.get("domain_L_pinn", 1.0)
        self.domain_D_pinn_env = env_config.get("domain_D_pinn", D_CHANNEL_FOR_PINN_ENV)
        x_coords = torch.linspace(0, self.domain_L_pinn, self.nx_obs, device=DEVICE)
        y_coords = torch.linspace(PINN_ENV_Y_MIN_CFG, self.domain_D_pinn_env, self.ny_obs, device=DEVICE)
        self.grid_x_obs_torch, self.grid_y_obs_torch = torch.meshgrid(x_coords, y_coords, indexing="ij")
        self.obs_spatial_coords_flat = torch.stack([self.grid_x_obs_torch.flatten(), self.grid_y_obs_torch.flatten()], dim=1)
        self.grid_x_obs_np_cpu = self.grid_x_obs_torch.cpu().numpy()
        self.grid_y_obs_np_cpu = self.grid_y_obs_torch.cpu().numpy()

        self.current_time = 0.0
        self.current_control_action = np.zeros(self.num_control_params, dtype=np.float32)
        self.initial_control_action = env_config.get("initial_control_action", np.zeros(self.num_control_params, dtype=np.float32))

    def _get_pinn_prediction_uvp(self, time_val, control_action_np):
        control_action_torch = torch.tensor(control_action_np, dtype=torch.float32, device=DEVICE)
        num_spatial_points = self.obs_spatial_coords_flat.shape[0]
        time_tensor = torch.full((num_spatial_points, 1), time_val, dtype=torch.float32, device=DEVICE)
        expanded_control_action = control_action_torch.unsqueeze(0).repeat(num_spatial_points, 1)
        pinn_input = torch.cat([self.obs_spatial_coords_flat, time_tensor, expanded_control_action], dim=1)
        with torch.no_grad():
            uvp_flat = self.pinn_model(pinn_input)
        return uvp_flat.reshape(self.nx_obs, self.ny_obs, 3).cpu().numpy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0.0
        self.current_step_env = 0
        self.current_control_action = np.copy(self.initial_control_action)
        obs = self._get_pinn_prediction_uvp(self.current_time, self.current_control_action)
        info = {}
        return obs, info

    def step(self, action_input):
        self.current_control_action = np.array(action_input, dtype=np.float32)
        self.current_time += self.dt
        self.current_step_env += 1
        current_uvp_field = self._get_pinn_prediction_uvp(self.current_time, self.current_control_action)
        action_penalty = np.sum(self.current_control_action**2) * 0.01
        reward = -action_penalty
        done = self.current_time >= (self.T_total - 1e-5)
        truncated = self.current_step_env >= self.max_steps_env
        if truncated and not done:
            done = True
        info = {"mse_info_action_penalty": action_penalty, "control_action": np.copy(self.current_control_action)}
        return current_uvp_field, reward, done, truncated, info

# --- 간단한 Gym 환경 정의 (Standalone PPO용) ---
class SimpleControlEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    def __init__(self, env_config):
        super().__init__()
        self.max_steps = env_config.get("max_episode_steps", 200)
        self.current_step = 0
        self.observation_dim = env_config.get("observation_dim", 4)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.observation_dim,), dtype=np.float32)
        self.action_dim = env_config.get("action_dim", 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.state = np.zeros(self.observation_dim, dtype=np.float32)
        self.target_state = np.array(env_config.get("target_state", [0.0] * self.observation_dim), dtype=np.float32)
        self.action_scale = env_config.get("action_scale", 0.1)
        self.state_decay = env_config.get("state_decay", 0.01)
        self.noise_level = env_config.get("noise_level", 0.02)
        self.reward_action_penalty_scale = env_config.get("reward_action_penalty_scale", -0.01)
        self.dt_sim_for_plot = env_config.get("dt_sim_for_plot", 0.01)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.5, high=0.5, size=self.observation_dim).astype(np.float32)
        self.current_step = 0
        info = {}
        return self.state, info

    def step(self, action):
        if isinstance(action, (np.ndarray, list)) and self.action_dim == 1:
            action_value = action[0]
        elif isinstance(action, (int, float)) and self.action_dim == 1:
             action_value = action
        else:
            action_value = action

        noise = self.np_random.normal(0, self.noise_level, size=self.observation_dim)
        action_effect = np.zeros(self.observation_dim, dtype=np.float32)
        if self.action_dim == 1:
            action_effect[:] = action_value * self.action_scale
        else:
            for i in range(min(self.action_dim, self.observation_dim)):
                action_effect[i] = action_value[i] * self.action_scale
        
        self.state = self.state * (1 - self.state_decay) + action_effect + noise
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        self.current_step += 1
        
        action_penalty_value = np.sum(np.array(action)**2)
        reward = self.reward_action_penalty_scale * action_penalty_value
        
        done = self.current_step >= self.max_steps
        truncated = False
        distance_to_target_info = np.linalg.norm(self.state - self.target_state)
        info = {
            "action_penalty": action_penalty_value,
            "distance_to_target": distance_to_target_info,
            "raw_action": np.copy(action)
        }
        return self.state, reward, done, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, State: {np.round(self.state, 2)}, Target (참고용): {self.target_state}")
    def close(self):
        pass

# --- 시뮬레이션 실행 및 데이터 수집 함수 ---
def run_pinn_ppo_episode(env_instance, ppo_model, deterministic_ppo=True):
    obs_history, action_history, reward_history, mse_info_history = [], [], [], []
    obs_env, _ = env_instance.reset()
    obs_for_ppo = obs_env[:,:,:2]
    obs_history.append(obs_env.copy())
    cumulative_reward = 0.0
    for step_num in range(env_instance.max_steps_env):
        action_to_take, _ = ppo_model.predict(obs_for_ppo, deterministic=deterministic_ppo)
        obs_env, reward, done, truncated, info = env_instance.step(action_to_take)
        obs_for_ppo = obs_env[:,:,:2]
        obs_history.append(obs_env.copy())
        action_history.append(info.get("control_action", np.copy(action_to_take)))
        reward_history.append(reward)
        mse_info_history.append(info.get("mse_info_action_penalty", float('inf')))
        cumulative_reward += reward
        if done or truncated: break
    final_mse_info_val = mse_info_history[-1] if mse_info_history and np.isfinite(mse_info_history[-1]) else float('inf')
    print(f"  PINN-PPO Sim. 총 보상: {cumulative_reward:.4f}, 최종 정보 값(액션 페널티): {final_mse_info_val:.4e}, 스텝 수: {len(action_history)}")
    return np.array(obs_history), np.array(action_history), np.array(mse_info_history), cumulative_reward

def run_simple_env_ppo_episode(env_instance, ppo_model, deterministic_ppo=True):
    obs_history, action_history, reward_history, distance_history = [], [], [], []
    obs, _ = env_instance.reset()
    obs_history.append(obs.copy())
    cumulative_reward = 0.0
    for _ in range(env_instance.max_steps):
        action_to_take, _ = ppo_model.predict(obs, deterministic=deterministic_ppo)
        obs, reward, done, truncated, info = env_instance.step(action_to_take)
        obs_history.append(obs.copy())
        action_history.append(info.get("raw_action", np.copy(action_to_take)))
        reward_history.append(reward)
        distance_history.append(info.get("distance_to_target", float('inf')))
        cumulative_reward += reward
        if done or truncated: break
    final_dist = distance_history[-1] if distance_history and np.isfinite(distance_history[-1]) else float('inf')
    print(f"  Standalone PPO Sim. 총 보상: {cumulative_reward:.3f}, 최종 거리 (참고용): {final_dist:.3f}, 스텝 수: {len(action_history)}")
    return np.array(obs_history), np.array(action_history), np.array(distance_history), cumulative_reward

# --- 시각화 함수들 ---
def plot_pinn_fields_comparison(
    obs_main_scenario, obs_baseline, 
    grid_x_np, grid_y_np, 
    time_step_index, dt_sim, 
    color_legends_dict, 
    main_scenario_title="Scenario 1", 
    baseline_scenario_title="Scenario 2" # 사용하지 않을 경우 None 또는 다른 이름
):
    field_names_map = {"u-velocity": 0, "v-velocity": 1, "pressure (p)": 2}
    
    scenarios_to_plot_data = {}
    if obs_main_scenario is not None and obs_main_scenario.shape[0] > 0 and time_step_index < obs_main_scenario.shape[0]:
        scenarios_to_plot_data[main_scenario_title] = obs_main_scenario[time_step_index]
    
    if obs_baseline is not None and obs_baseline.shape[0] > 0 and time_step_index < obs_baseline.shape[0]:
        scenarios_to_plot_data[baseline_scenario_title] = obs_baseline[time_step_index]

    if not scenarios_to_plot_data:
        print(f"시각화할 데이터 없음 (시간 인덱스: {time_step_index}).")
        return

    num_rows = len(scenarios_to_plot_data)
    time_val = time_step_index * dt_sim
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, 5 * num_rows), squeeze=False)
    # 전체 그림 제목 설정
    if num_rows > 1:
        fig.suptitle(f'Field Comparison at Time = {time_val:.3f}s (Time Step Index: {time_step_index})', fontsize=16)
    elif num_rows == 1: # 단일 시나리오의 경우, 해당 시나리오 이름을 제목에 포함
        single_scenario_name = list(scenarios_to_plot_data.keys())[0]
        fig.suptitle(f'{single_scenario_name} - Fields at Time = {time_val:.3f}s (T_idx: {time_step_index})', fontsize=16)


    row_idx = 0
    for scenario_name_key, current_obs_uvp in scenarios_to_plot_data.items():
        for col_idx, (field_plot_name, field_data_idx) in enumerate(field_names_map.items()):
            ax = axes[row_idx, col_idx]
            data_to_plot = current_obs_uvp[:, :, field_data_idx]
            
            vmin, vmax = None, None
            effective_legend = color_legends_dict.get(field_plot_name) if color_legends_dict else None
            if effective_legend is not None:
                vmin, vmax = effective_legend
            else: 
                data_min_val, data_max_val = np.min(data_to_plot), np.max(data_to_plot)
                if data_min_val == data_max_val: 
                    data_min_val -= 0.1 if data_min_val != 0 else 0.0
                    data_max_val += 0.1 if data_max_val != 0 else 0.1
                    if data_min_val == data_max_val :
                        data_min_val = -0.1; data_max_val = 0.1
                vmin, vmax = data_min_val, data_max_val

            contour = ax.contourf(grid_x_np, grid_y_np, data_to_plot.T, levels=50, cmap='jet', vmin=vmin, vmax=vmax)
            fig.colorbar(contour, ax=ax)
            # subplot 제목은 필드 이름만으로 변경 (전체 시나리오 이름은 fig.suptitle에 있음)
            ax.set_title(field_plot_name if num_rows > 1 else f"{scenario_name_key} - {field_plot_name}")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.axis('square')
        row_idx += 1
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95 if num_rows > 1 else 0.92]) # suptitle 공간 확보
    if SAVE_PLOTS:
        # 파일 이름에 메인 시나리오 제목을 포함하여 구분
        filename_suffix = main_scenario_title.replace(" ", "_").replace(":", "").replace("-", "_")
        save_path = os.path.join(PLOT_SAVE_DIR, f"fields_t_idx{time_step_index}_{filename_suffix}.png")
        plt.savefig(save_path); print(f"  필드 그림 저장: {save_path}")
    plt.show(block=False)
    plt.pause(0.1)

def plot_standalone_ppo_states(obs_standalone_hist, dt_simulation, state_dim):
    if obs_standalone_hist is None or obs_standalone_hist.shape[0] == 0:
        print("Standalone PPO 상태 데이터가 없어 플롯을 생성할 수 없습니다.")
        return
    num_steps = obs_standalone_hist.shape[0]
    time_axis = np.arange(num_steps) * dt_simulation
    plt.figure(figsize=(12, min(15, 3 * state_dim)))
    plt.suptitle("State Variable Evolution (Standalone PPO)", fontsize=16)
    for i in range(state_dim):
        plt.subplot(state_dim, 1, i + 1)
        plt.plot(time_axis, obs_standalone_hist[:, i], label=f'State[{i}]')
        plt.ylabel(f'State[{i}]'); plt.grid(True, linestyle=':')
        if i == state_dim -1 : plt.xlabel("Time (s)")
        plt.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if SAVE_PLOTS:
        plt.savefig(os.path.join(PLOT_SAVE_DIR, "standalone_ppo_state_evolution.png"))
        print(f"  Standalone PPO 상태 변수 플롯 저장 완료.")
    plt.show(block=False)
    plt.pause(0.1)

def plot_ppo_actions_comparison(actions_pinn_ppo, dt_pinn_ppo, actions_standalone_ppo, dt_standalone_ppo):
    num_action_dims = 0
    if actions_pinn_ppo is not None and actions_pinn_ppo.shape[0] > 0:
        num_action_dims = actions_pinn_ppo.shape[1] if actions_pinn_ppo.ndim > 1 else 1
    elif actions_standalone_ppo is not None and actions_standalone_ppo.shape[0] > 0:
        num_action_dims = actions_standalone_ppo.shape[1] if actions_standalone_ppo.ndim > 1 else 1
    if num_action_dims == 0:
        print("플롯할 제어 액션 데이터가 없습니다."); return

    fig, axes = plt.subplots(num_action_dims, 1, figsize=(12, min(15, 5 * num_action_dims)), squeeze=False)
    fig.suptitle("PPO Control Action Comparison", fontsize=16)
    axes = axes.flatten()
    for i in range(num_action_dims):
        ax = axes[i]
        plot_pinn = False
        if actions_pinn_ppo is not None and actions_pinn_ppo.shape[0] > 0:
            time_axis_pinn = np.arange(actions_pinn_ppo.shape[0]) * dt_pinn_ppo
            action_data_pinn = actions_pinn_ppo[:,i] if actions_pinn_ppo.ndim > 1 and actions_pinn_ppo.shape[1] > i else actions_pinn_ppo
            ax.plot(time_axis_pinn, action_data_pinn, label=f'PINN-PPO Action[{i}]', color='royalblue', alpha=0.8)
            plot_pinn = True
        plot_standalone = False
        if actions_standalone_ppo is not None and actions_standalone_ppo.shape[0] > 0:
            time_axis_standalone = np.arange(actions_standalone_ppo.shape[0]) * dt_standalone_ppo
            action_data_standalone = actions_standalone_ppo[:,i] if actions_standalone_ppo.ndim > 1 and actions_standalone_ppo.shape[1] > i else actions_standalone_ppo
            ax.plot(time_axis_standalone, action_data_standalone, label=f'Standalone PPO Action[{i}]', color='orangered', linestyle='--', alpha=0.8)
            plot_standalone = True
        if plot_pinn or plot_standalone:
            ax.set_ylabel(f"Action[{i}]"); ax.legend(loc='upper right'); ax.grid(True, linestyle=':')
            if i == num_action_dims -1 : ax.set_xlabel("Time (s)")
            ax.set_title(f"Control Action Dimension {i}")
        else: ax.axis('off')
    plt.tight_layout(rect=[0,0,1,0.95])
    if SAVE_PLOTS:
        plt.savefig(os.path.join(PLOT_SAVE_DIR, "ppo_actions_comparison.png"))
        print(f"  PPO 액션 비교 플롯 저장 완료.")
    plt.show(block=False)
    plt.pause(0.1)

# --- 메인 실행 ---
if __name__ == "__main__":
    # --- PINN-PPO 환경 설정 및 실행 ---
    print("--- PINN-PPO 시나리오 평가 ---")
    nx_pinn_vis, ny_pinn_vis = 20, 20
    env_config_pinn_ppo = {
        "T_sim": 0.2, "dt_sim": 1e-2, "num_control_params": 1,
        "action_low": -1.0, "action_high": 1.0,
        "nx_obs": nx_pinn_vis, "ny_obs": ny_pinn_vis,
        "domain_L_pinn": 1.0, "domain_D_pinn": D_CHANNEL_FOR_PINN_ENV,
        "initial_control_action": np.array([0.0], dtype=np.float32)
    }
    obs_pinn_hist_data, actions_pinn_hist_data, mses_info_pinn_hist, cum_rew_pinn_data = None, None, None, None
    env_pinn_ppo_vis_for_plot = None

    if os.path.exists(PINN_PPO_POLICY_FILE_PATH) and os.path.exists(PRETRAINED_PINN_MODEL_PATH):
        pinn_ppo_model_loaded = PPO.load(PINN_PPO_POLICY_FILE_PATH, device=DEVICE)
        print(f"학습된 PINN-PPO 정책 로드 완료: {PINN_PPO_POLICY_FILE_PATH}")
        env_pinn_ppo_vis = PINNNavierStokesEnv(PRETRAINED_PINN_MODEL_PATH, env_config_pinn_ppo)
        env_pinn_ppo_vis_for_plot = env_pinn_ppo_vis
        obs_pinn_hist_data, actions_pinn_hist_data, mses_info_pinn_hist, cum_rew_pinn_data = \
            run_pinn_ppo_episode(env_pinn_ppo_vis, ppo_model=pinn_ppo_model_loaded)
    else:
        print(f"오류: PINN-PPO 실행에 필요한 파일({PINN_PPO_POLICY_FILE_PATH} 또는 {PRETRAINED_PINN_MODEL_PATH})을 찾을 수 없습니다.")

    # --- Standalone PPO 환경 설정 및 실행 ---
    print("\n--- Standalone PPO 시나리오 평가 ---")
    simple_env_config_vis = {
        "max_episode_steps": int(env_config_pinn_ppo["T_sim"] / env_config_pinn_ppo["dt_sim"]),
        "observation_dim": 4, "action_dim": 1,
        "target_state": [0.0] * 4,
        "action_scale": 0.2, "state_decay": 0.01, "noise_level": 0.05,
        "reward_action_penalty_scale": -0.01,
        "dt_sim_for_plot": env_config_pinn_ppo["dt_sim"]
    }
    obs_standalone_hist_data, actions_standalone_hist_data, dist_standalone_hist, cum_rew_standalone_data = None, None, None, None

    if os.path.exists(STANDALONE_PPO_POLICY_FILE_PATH):
        standalone_ppo_model_loaded = PPO.load(STANDALONE_PPO_POLICY_FILE_PATH, device=DEVICE)
        print(f"학습된 Standalone PPO 정책 로드 완료: {STANDALONE_PPO_POLICY_FILE_PATH}")
        env_standalone_vis = SimpleControlEnv(simple_env_config_vis)
        obs_standalone_hist_data, actions_standalone_hist_data, dist_standalone_hist, cum_rew_standalone_data = \
            run_simple_env_ppo_episode(env_standalone_vis, ppo_model=standalone_ppo_model_loaded)
    else:
        print(f"오류: Standalone PPO 정책 파일({STANDALONE_PPO_POLICY_FILE_PATH})을 찾을 수 없습니다.")

    # --- Standalone PPO 액션으로 PINN 유동장 생성 ---
    obs_pinn_driven_by_standalone_actions = None
    if actions_standalone_hist_data is not None and env_pinn_ppo_vis_for_plot is not None:
        print("\n--- Standalone PPO 액션을 사용하여 PINN 유동장 생성 중 ---")
        pinn_driven_fields_list = []
        current_time_for_pinn = 0.0
        pinn_predictor_env = env_pinn_ppo_vis_for_plot
        for i in range(len(actions_standalone_hist_data)):
            action_from_standalone = actions_standalone_hist_data[i]
            if isinstance(action_from_standalone, (int, float)):
                control_action_for_pinn = np.array([action_from_standalone], dtype=np.float32)
            elif isinstance(action_from_standalone, np.ndarray) and action_from_standalone.ndim == 0:
                control_action_for_pinn = np.array([action_from_standalone.item()], dtype=np.float32)
            else:
                control_action_for_pinn = np.array(action_from_standalone, dtype=np.float32)
            control_action_for_pinn = np.clip(control_action_for_pinn, pinn_predictor_env.action_space.low, pinn_predictor_env.action_space.high)
            uvp_field = pinn_predictor_env._get_pinn_prediction_uvp(current_time_for_pinn, control_action_for_pinn)
            pinn_driven_fields_list.append(uvp_field)
            current_time_for_pinn += env_config_pinn_ppo["dt_sim"]
        obs_pinn_driven_by_standalone_actions = np.array(pinn_driven_fields_list)
        print(f"  Standalone PPO 액션으로 {len(pinn_driven_fields_list)}개의 PINN 유동장 생성 완료.")

    # --- 정량적 성능 지표 저장 및 출력 ---
    evaluation_summary_list = []
    if cum_rew_pinn_data is not None:
        pinn_metrics = {
            "Scenario": "PINN-PPO", "Cumulative Reward": cum_rew_pinn_data,
            "Avg Info Metric (Action Penalty)": np.mean(mses_info_pinn_hist[np.isfinite(mses_info_pinn_hist)][1:]) if mses_info_pinn_hist is not None and len(mses_info_pinn_hist[np.isfinite(mses_info_pinn_hist)]) > 1 else float('inf'),
            "Final Info Metric (Action Penalty)": mses_info_pinn_hist[-1] if mses_info_pinn_hist is not None and len(mses_info_pinn_hist) > 0 and np.isfinite(mses_info_pinn_hist[-1]) else float('inf'),
            "Total Control Effort (sum_a^2)": np.sum(actions_pinn_hist_data**2) if actions_pinn_hist_data is not None and actions_pinn_hist_data.size > 0 else 0.0 }
        evaluation_summary_list.append(pinn_metrics)
    if cum_rew_standalone_data is not None:
        standalone_metrics = {
            "Scenario": "Standalone PPO (on SimpleEnv)", "Cumulative Reward": cum_rew_standalone_data,
            "Avg Info Metric (Distance - 참고용)": np.mean(dist_standalone_hist[np.isfinite(dist_standalone_hist)][1:]) if dist_standalone_hist is not None and len(dist_standalone_hist[np.isfinite(dist_standalone_hist)]) > 1 else float('inf'),
            "Final Info Metric (Distance - 참고용)": dist_standalone_hist[-1] if dist_standalone_hist is not None and len(dist_standalone_hist) > 0 and np.isfinite(dist_standalone_hist[-1]) else float('inf'),
            "Total Control Effort (sum_a^2 in SimpleEnv)": np.sum(actions_standalone_hist_data**2) if actions_standalone_hist_data is not None and actions_standalone_hist_data.size > 0 else 0.0 }
        evaluation_summary_list.append(standalone_metrics)
    if evaluation_summary_list:
        df_evaluation_summary = pd.DataFrame(evaluation_summary_list)
        if not os.path.exists(os.path.dirname(EVALUATION_RESULTS_SAVE_PATH)):
             os.makedirs(os.path.dirname(EVALUATION_RESULTS_SAVE_PATH), exist_ok=True)
        try:
            df_evaluation_summary.to_csv(EVALUATION_RESULTS_SAVE_PATH, index=False, float_format='%.4e')
            print(f"\n--- 평가 결과 요약 CSV 저장 완료: {EVALUATION_RESULTS_SAVE_PATH} ---")
            print(df_evaluation_summary.to_string(float_format="%.4e"))
        except Exception as e: print(f"평가 결과 요약 CSV 저장 중 오류: {e}")
    else: print("\n생성된 평가 데이터가 없습니다.")
    print("--- ---")

    # --- 시각화 ---
    # 1. PINN with PPO 필드 시각화 (기존)
    if obs_pinn_hist_data is not None and obs_pinn_hist_data.shape[0] > 1:
        if env_pinn_ppo_vis_for_plot is not None:
            grid_x_pinn_plot = env_pinn_ppo_vis_for_plot.grid_x_obs_np_cpu
            grid_y_pinn_plot = env_pinn_ppo_vis_for_plot.grid_y_obs_np_cpu
            time_indices_pinn_plot = [0, obs_pinn_hist_data.shape[0] // 2, obs_pinn_hist_data.shape[0] - 1]
            time_indices_pinn_plot = [idx for idx in time_indices_pinn_plot if idx < obs_pinn_hist_data.shape[0]]
            for t_idx_plot in time_indices_pinn_plot:
                print(f"\n--- [기존] PINN with PPO 유동장 (Time Step Index: {t_idx_plot}) ---")
                plot_pinn_fields_comparison(
                    obs_pinn_hist_data, None,
                    grid_x_pinn_plot, grid_y_pinn_plot,
                    time_step_index=t_idx_plot, dt_sim=env_config_pinn_ppo["dt_sim"],
                    color_legends_dict=FIELD_COLOR_LEGENDS,
                    main_scenario_title="PINN with PPO" # 수정된 함수에 맞게 제목 전달
                )
        else: print("\nPINN-PPO 환경 인스턴스가 없어 [기존] 필드 시각화를 생성할 수 없습니다.")
    else: print("\n[기존] PINN-PPO 시뮬레이션 결과 데이터가 부족하여 필드 시각화를 생성할 수 없습니다.")

    # 1.bis. Standalone PPO 액션으로 구동된 PINN 유동장 시각화
    if obs_pinn_driven_by_standalone_actions is not None and obs_pinn_driven_by_standalone_actions.shape[0] > 1:
        if env_pinn_ppo_vis_for_plot is not None:
            grid_x_pinn_plot = env_pinn_ppo_vis_for_plot.grid_x_obs_np_cpu
            grid_y_pinn_plot = env_pinn_ppo_vis_for_plot.grid_y_obs_np_cpu
            time_indices_driven_plot = [0, obs_pinn_driven_by_standalone_actions.shape[0] // 2, obs_pinn_driven_by_standalone_actions.shape[0] - 1]
            time_indices_driven_plot = [idx for idx in time_indices_driven_plot if idx < obs_pinn_driven_by_standalone_actions.shape[0]]
            for t_idx_plot in time_indices_driven_plot:
                print(f"\n--- [추가] Standalone PPO Actions (Time Step Index: {t_idx_plot}) ---")
                plot_pinn_fields_comparison( 
                    obs_pinn_driven_by_standalone_actions, None,
                    grid_x_pinn_plot, grid_y_pinn_plot,
                    time_step_index=t_idx_plot, dt_sim=env_config_pinn_ppo["dt_sim"],
                    color_legends_dict=FIELD_COLOR_LEGENDS,
                    main_scenario_title="Standalone PPO Actions" # 수정된 함수에 맞게 제목 전달
                )
        else: print("\nPINN-PPO 환경 인스턴스가 없어 [추가] 필드 시각화를 생성할 수 없습니다.")
    else: print("\n[추가] Standalone PPO 액션으로 구동된 PINN 유동장 데이터가 없어 시각화를 생성할 수 없습니다.")

    # 2. Standalone PPO 상태 변수 시각화 (기존)
    if obs_standalone_hist_data is not None and obs_standalone_hist_data.shape[0] > 1:
        print("\n--- Standalone PPO 상태 변수 시각화 ---")
        plot_standalone_ppo_states(
            obs_standalone_hist_data,
            dt_simulation=simple_env_config_vis["dt_sim_for_plot"],
            state_dim=simple_env_config_vis["observation_dim"]
        )
    else: print("\nStandalone PPO 시뮬레이션 결과 데이터가 부족하여 상태 시각화를 생성할 수 없습니다.")
        
    # 3. 두 PPO 시나리오의 제어 액션 비교 시각화 (기존)
    print("\n--- PPO 제어 액션 비교 시각화 ---")
    plot_ppo_actions_comparison(
        actions_pinn_hist_data, env_config_pinn_ppo["dt_sim"],
        actions_standalone_hist_data, simple_env_config_vis["dt_sim_for_plot"]
    )

    print("\n모든 시각화 및 평가 시도 완료. 창을 닫으려면 모든 플롯 창을 닫아주세요.")
    if SAVE_PLOTS:
        print(f"모든 플롯은 '{PLOT_SAVE_DIR}' 디렉토리에 저장되었습니다.")
    plt.show()