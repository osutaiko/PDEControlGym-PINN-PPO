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
# PINN with PPO 관련 경로
PRETRAINED_PINN_MODEL_PATH = os.path.join(ROOT_DIR, "time_control_pinn.pth") # 사용자 모델 경로
PINN_PPO_POLICY_FILE_PATH = os.path.join(ROOT_DIR, "ppo_pinn_control_policy.zip") # 사용자 정책 경로

# Standalone PPO 관련 경로
STANDALONE_PPO_POLICY_FILE_PATH = os.path.join(ROOT_DIR, "ppo_standalone_policy.zip") # ppo_standalone_trainer.py 에서 저장한 파일

EVALUATION_RESULTS_SAVE_PATH = "./final_comparison_log/ppo_scenario_comparison.csv"
PLOT_SAVE_DIR = "./final_comparison_log/"
if not os.path.exists(PLOT_SAVE_DIR):
    os.makedirs(PLOT_SAVE_DIR)
SAVE_PLOTS = True # 플롯 저장 여부

# --- 물리 상수 (PINN 환경용) ---
D_CHANNEL_FOR_PINN_ENV = 1.0
PINN_ENV_Y_MIN_CFG = 0.0
PINN_ENV_Y_MAX_CFG = D_CHANNEL_FOR_PINN_ENV # PINN 환경의 Y 최대값

FIELD_COLOR_LEGENDS = {
    "u-velocity": [0, 1.5],
    "v-velocity": [-0.3, 0.3],
    "pressure (p)": None # 자동 스케일링
}

# --- PINN 모델 정의 (PINN-PPO용) ---
class TimeControlPINN_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_layers = 5
        neurons_per_layer = 128
        layers = [input_dim] + [neurons_per_layer] * hidden_layers + [3] # Output: u, v, p
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )
        self.activation = torch.tanh
    def forward(self, xytc): # x, y, time, control_params
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

        pinn_input_dim = 2 + 1 + self.num_control_params # x, y, t, c
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
        self.current_control_action = np.array(action_input, dtype=np.float32) # Ensure action is np.float32
        self.current_time += self.dt
        self.current_step_env += 1
        current_uvp_field = self._get_pinn_prediction_uvp(self.current_time, self.current_control_action)

        action_penalty = np.sum(self.current_control_action**2) * 0.01
        reward = -action_penalty # 보상은 액션 페널티만으로 구성

        done = self.current_time >= (self.T_total - 1e-5) # 부동소수점 비교 주의
        truncated = self.current_step_env >= self.max_steps_env
        if truncated and not done: # max_steps에 도달하면 done으로 간주
            done = True

        info = {"mse_info_action_penalty": action_penalty, "control_action": np.copy(self.current_control_action)}
        return current_uvp_field, reward, done, truncated, info

# --- Gym 환경 정의 (Standalone PPO용) ---
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
        # target_state는 정의는 되어 있지만, 수정된 보상 함수에서는 직접적인 목표로 사용되지 않음
        self.target_state = np.array(env_config.get("target_state", [0.0] * self.observation_dim), dtype=np.float32)
        
        self.action_scale = env_config.get("action_scale", 0.1)
        self.state_decay = env_config.get("state_decay", 0.01)
        self.noise_level = env_config.get("noise_level", 0.02)
        
        # reward_distance_scale은 이제 사용되지 않음
        self.reward_action_penalty_scale = env_config.get("reward_action_penalty_scale", -0.01)
        self.dt_sim_for_plot = env_config.get("dt_sim_for_plot", 0.01)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.5, high=0.5, size=self.observation_dim).astype(np.float32)
        self.current_step = 0
        info = {}
        return self.state, info

    def step(self, action):
        # 입력 action이 스칼라일 수도, 배열일 수도 있으므로 처리
        if isinstance(action, (np.ndarray, list)) and self.action_dim == 1:
            action_value = action[0]
        elif isinstance(action, (int, float)) and self.action_dim == 1:
             action_value = action
        else: # 다차원 액션
            action_value = action # 이 경우 action은 np.ndarray 여야 함

        noise = self.np_random.normal(0, self.noise_level, size=self.observation_dim)
        action_effect = np.zeros(self.observation_dim, dtype=np.float32)

        if self.action_dim == 1:
            action_effect[:] = action_value * self.action_scale
        else:
            for i in range(min(self.action_dim, self.observation_dim)):
                action_effect[i] = action_value[i] * self.action_scale # 다차원 액션의 경우 action_value[i]
        
        self.state = self.state * (1 - self.state_decay) + action_effect + noise
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        self.current_step += 1
        
        action_penalty_value = np.sum(np.array(action)**2) # action을 np.array로 변환
        # 보상은 액션 페널티에 의해서만 결정 (PINNNavierStokesEnv와 동일한 목표 구조)
        reward = self.reward_action_penalty_scale * action_penalty_value
        
        done = self.current_step >= self.max_steps
        truncated = False # 이 환경에서는 truncated를 별도로 사용하지 않음

        distance_to_target_info = np.linalg.norm(self.state - self.target_state) # 참고용 정보
        info = {
            "action_penalty": action_penalty_value,
            "distance_to_target": distance_to_target_info,
            "raw_action": np.copy(action)
        }
        return self.state, reward, done, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, State: {np.round(self.state, 2)}, Target ( 참고용 ): {self.target_state}")
    def close(self):
        pass

# --- 시뮬레이션 실행 및 데이터 수집 함수 ---
def run_pinn_ppo_episode(env_instance, ppo_model, deterministic_ppo=True):
    obs_history, action_history, reward_history, mse_info_history = [], [], [], []
    obs_env, _ = env_instance.reset() # obs_env는 (nx,ny,3) 형태

    # PPO 모델은 (nx,ny,2) 형태의 관찰을 예상 (u,v 필드)
    obs_for_ppo = obs_env[:,:,:2]
    
    obs_history.append(obs_env.copy()) # 전체 관찰(u,v,p)은 히스토리에 저장
    cumulative_reward = 0.0

    for step_num in range(env_instance.max_steps_env):
        action_to_take, _ = ppo_model.predict(obs_for_ppo, deterministic=deterministic_ppo)
        obs_env, reward, done, truncated, info = env_instance.step(action_to_take)
        
        obs_for_ppo = obs_env[:,:,:2] # 다음 스텝을 위한 PPO 입력도 슬라이싱
        
        obs_history.append(obs_env.copy())
        action_history.append(info.get("control_action", np.copy(action_to_take)))
        reward_history.append(reward)
        mse_info_history.append(info.get("mse_info_action_penalty", float('inf')))
        cumulative_reward += reward
        if done or truncated:
            break
            
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
        if done or truncated:
            break
            
    final_dist = distance_history[-1] if distance_history and np.isfinite(distance_history[-1]) else float('inf')
    print(f"  Standalone PPO Sim. 총 보상: {cumulative_reward:.3f}, 최종 거리 (참고용): {final_dist:.3f}, 스텝 수: {len(action_history)}")
    return np.array(obs_history), np.array(action_history), np.array(distance_history), cumulative_reward

# --- 시각화 함수들 ---
def plot_pinn_fields_comparison(obs_pinn_ppo, obs_baseline, grid_x_np, grid_y_np, time_step_index, dt_sim, color_legends_dict):
    field_names_map = {"u-velocity": 0, "v-velocity": 1, "pressure (p)": 2}
    num_rows = 1
    scenario_data = {"PINN-PPO": obs_pinn_ppo}
    if obs_baseline is not None:
        num_rows = 2
        scenario_data["Baseline"] = obs_baseline

    if obs_pinn_ppo is None or obs_pinn_ppo.shape[0] == 0:
        print("PINN-PPO 시뮬레이션 데이터가 없어 필드 플롯을 생성할 수 없습니다.")
        return
    if time_step_index >= obs_pinn_ppo.shape[0]:
        print(f"요청된 시간 인덱스({time_step_index})가 PINN-PPO 데이터 길이({obs_pinn_ppo.shape[0]})를 초과합니다.")
        return
    if obs_baseline is not None and time_step_index >= obs_baseline.shape[0]:
        print(f"요청된 시간 인덱스({time_step_index})가 Baseline 데이터 길이({obs_baseline.shape[0]})를 초과합니다.")
        return

    time_val = time_step_index * dt_sim
    # If obs_baseline is None, num_rows should be 1 for the PINN-PPO data only
    # The original plot_pinn_ppo_fields was for a single scenario
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # Plotting only PINN-PPO
    fig.suptitle(f'PINN-PPO Fields at Time = {time_val:.3f}s (Time Step Index: {time_step_index})', fontsize=16)
    
    current_obs_uvp = obs_pinn_ppo[time_step_index]
    for col_idx, (name, field_idx) in enumerate(field_names_map.items()):
        ax = axes[col_idx]
        data_to_plot = current_obs_uvp[:, :, field_idx]
        
        vmin, vmax = None, None
        effective_legend = color_legends_dict.get(name) if color_legends_dict else None
        if effective_legend is not None:
            vmin, vmax = effective_legend
        else: # 자동 스케일링
            data_min, data_max = np.min(data_to_plot), np.max(data_to_plot)
            if data_min == data_max: # 모든 값이 같을 경우 대비
                data_min -= 0.1 if data_min != 0 else 0.0
                data_max += 0.1 if data_max != 0 else 0.1
                if data_min == data_max : # Still same (e.g. all zeros)
                    data_min = -0.1
                    data_max = 0.1
            vmin, vmax = data_min, data_max

        contour = ax.contourf(grid_x_np, grid_y_np, data_to_plot.T, levels=50, cmap='jet', vmin=vmin, vmax=vmax) # .T for (Ny,Nx) data with (x,y) coords
        fig.colorbar(contour, ax=ax)
        ax.set_title(f"PINN-PPO - {name}")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.axis('square')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if SAVE_PLOTS:
        save_path = os.path.join(PLOT_SAVE_DIR, f"pinn_ppo_fields_t_idx{time_step_index}.png")
        plt.savefig(save_path); print(f"  PINN-PPO 필드 그림 저장: {save_path}")
    plt.show(block=False)
    plt.pause(0.1)


def plot_standalone_ppo_states(obs_standalone_hist, dt_simulation, state_dim):
    if obs_standalone_hist is None or obs_standalone_hist.shape[0] == 0:
        print("Standalone PPO 상태 데이터가 없어 플롯을 생성할 수 없습니다.")
        return
    num_steps = obs_standalone_hist.shape[0]
    time_axis = np.arange(num_steps) * dt_simulation
    
    plt.figure(figsize=(12, min(15, 3 * state_dim))) # Max height 15
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
        else: # 데이터가 없는 경우 subplot 비활성화
            ax.axis('off')
            
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
    nx_pinn_vis, ny_pinn_vis = 20, 20 # PINN 환경의 관찰 격자 크기
    env_config_pinn_ppo = {
        "T_sim": 0.2, "dt_sim": 1e-2, "num_control_params": 1,
        "action_low": -1.0, "action_high": 1.0,
        "nx_obs": nx_pinn_vis, "ny_obs": ny_pinn_vis,
        "domain_L_pinn": 1.0, "domain_D_pinn": D_CHANNEL_FOR_PINN_ENV,
        "initial_control_action": np.array([0.0], dtype=np.float32)
    }
    obs_pinn_hist_data, actions_pinn_hist_data, mses_info_pinn_hist, cum_rew_pinn_data = None, None, None, None
    env_pinn_ppo_vis_for_plot = None # 플로팅 시 env 인스턴스 접근용

    if os.path.exists(PINN_PPO_POLICY_FILE_PATH) and os.path.exists(PRETRAINED_PINN_MODEL_PATH):
        pinn_ppo_model_loaded = PPO.load(PINN_PPO_POLICY_FILE_PATH, device=DEVICE)
        print(f"학습된 PINN-PPO 정책 로드 완료: {PINN_PPO_POLICY_FILE_PATH}")
        env_pinn_ppo_vis = PINNNavierStokesEnv(PRETRAINED_PINN_MODEL_PATH, env_config_pinn_ppo)
        env_pinn_ppo_vis_for_plot = env_pinn_ppo_vis # Save instance for plotting grids
        obs_pinn_hist_data, actions_pinn_hist_data, mses_info_pinn_hist, cum_rew_pinn_data = \
            run_pinn_ppo_episode(env_pinn_ppo_vis, ppo_model=pinn_ppo_model_loaded)
    else:
        print(f"오류: PINN-PPO 실행에 필요한 파일({PINN_PPO_POLICY_FILE_PATH} 또는 {PRETRAINED_PINN_MODEL_PATH})을 찾을 수 없습니다.")

    # --- Standalone PPO 환경 설정 및 실행 ---
    print("\n--- Standalone PPO 시나리오 평가 ---")
    simple_env_config_vis = {
        "max_episode_steps": int(env_config_pinn_ppo["T_sim"] / env_config_pinn_ppo["dt_sim"]),
        "observation_dim": 4, "action_dim": 1,
        "target_state": [0.0] * 4, # 보상에는 사용되지 않지만 참고용으로 유지
        "action_scale": 0.2, "state_decay": 0.01, "noise_level": 0.05,
        "reward_action_penalty_scale": -0.01, # PINNNavierStokesEnv와 유사한 페널티 스케일
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

    # --- 정량적 성능 지표 저장 및 출력 ---
    evaluation_summary_list = []
    if cum_rew_pinn_data is not None:
        pinn_metrics = {
            "Scenario": "PINN-PPO", 
            "Cumulative Reward": cum_rew_pinn_data,
            "Avg Info Metric (Action Penalty)": np.mean(mses_info_pinn_hist[np.isfinite(mses_info_pinn_hist)][1:]) if mses_info_pinn_hist is not None and len(mses_info_pinn_hist[np.isfinite(mses_info_pinn_hist)]) > 1 else float('inf'),
            "Final Info Metric (Action Penalty)": mses_info_pinn_hist[-1] if mses_info_pinn_hist is not None and len(mses_info_pinn_hist) > 0 and np.isfinite(mses_info_pinn_hist[-1]) else float('inf'),
            "Total Control Effort (sum_a^2)": np.sum(actions_pinn_hist_data**2) if actions_pinn_hist_data is not None and actions_pinn_hist_data.size > 0 else 0.0
        }
        evaluation_summary_list.append(pinn_metrics)
    
    if cum_rew_standalone_data is not None:
        standalone_metrics = {
            "Scenario": "Standalone PPO", 
            "Cumulative Reward": cum_rew_standalone_data,
            "Avg Info Metric (Distance - 참고용)": np.mean(dist_standalone_hist[np.isfinite(dist_standalone_hist)][1:]) if dist_standalone_hist is not None and len(dist_standalone_hist[np.isfinite(dist_standalone_hist)]) > 1 else float('inf'),
            "Final Info Metric (Distance - 참고용)": dist_standalone_hist[-1] if dist_standalone_hist is not None and len(dist_standalone_hist) > 0 and np.isfinite(dist_standalone_hist[-1]) else float('inf'),
            "Total Control Effort (sum_a^2)": np.sum(actions_standalone_hist_data**2) if actions_standalone_hist_data is not None and actions_standalone_hist_data.size > 0 else 0.0
        }
        evaluation_summary_list.append(standalone_metrics)
    
    if evaluation_summary_list:
        df_evaluation_summary = pd.DataFrame(evaluation_summary_list)
        if not os.path.exists(os.path.dirname(EVALUATION_RESULTS_SAVE_PATH)):
             os.makedirs(os.path.dirname(EVALUATION_RESULTS_SAVE_PATH), exist_ok=True)
        try:
            df_evaluation_summary.to_csv(EVALUATION_RESULTS_SAVE_PATH, index=False, float_format='%.4e')
            print(f"\n--- 평가 결과 요약 CSV 저장 완료: {EVALUATION_RESULTS_SAVE_PATH} ---")
            print(df_evaluation_summary.to_string(float_format="%.4e"))
        except Exception as e:
            print(f"평가 결과 요약 CSV 저장 중 오류: {e}")
    else:
        print("\n생성된 평가 데이터가 없습니다.")
    print("--- ---")

    # --- 시각화 ---
    # 1. PINN with PPO 필드 시각화
    if obs_pinn_hist_data is not None and obs_pinn_hist_data.shape[0] > 1:
        if env_pinn_ppo_vis_for_plot is not None: # PINN Env 인스턴스가 성공적으로 생성되었는지 확인
            grid_x_pinn_plot = env_pinn_ppo_vis_for_plot.grid_x_obs_np_cpu
            grid_y_pinn_plot = env_pinn_ppo_vis_for_plot.grid_y_obs_np_cpu
            
            time_indices_pinn_plot = [0, obs_pinn_hist_data.shape[0] // 2, obs_pinn_hist_data.shape[0] - 1]
            # 유효한 인덱스만 필터링
            time_indices_pinn_plot = [idx for idx in time_indices_pinn_plot if idx < obs_pinn_hist_data.shape[0]]

            for t_idx_plot in time_indices_pinn_plot:
                print(f"\n--- PINN with PPO 유동장(u,v,p) 필드 (Time Step Index: {t_idx_plot}) ---")
                plot_pinn_fields_comparison(
                    obs_pinn_hist_data, None, 
                    grid_x_pinn_plot, grid_y_pinn_plot,
                    time_step_index=t_idx_plot, dt_sim=env_config_pinn_ppo["dt_sim"],
                    color_legends_dict=FIELD_COLOR_LEGENDS 
                )
        else:
            print("\nPINN-PPO 환경 인스턴스가 없어 필드 시각화를 생성할 수 없습니다.")
    else:
        print("\nPINN-PPO 시뮬레이션 결과 데이터가 부족하여 필드 시각화를 생성할 수 없습니다.")

    # 2. Standalone PPO 상태 변수 시각화
    if obs_standalone_hist_data is not None and obs_standalone_hist_data.shape[0] > 1:
        print("\n--- Standalone PPO 상태 변수 시각화 ---")
        plot_standalone_ppo_states(
            obs_standalone_hist_data, 
            dt_simulation=simple_env_config_vis["dt_sim_for_plot"],
            state_dim=simple_env_config_vis["observation_dim"]
        )
    else:
        print("\nStandalone PPO 시뮬레이션 결과 데이터가 부족하여 상태 시각화를 생성할 수 없습니다.")
        
    # 3. 두 PPO 시나리오의 제어 액션 비교 시각화
    print("\n--- PPO 제어 액션 비교 시각화 ---")
    plot_ppo_actions_comparison(
        actions_pinn_hist_data, env_config_pinn_ppo["dt_sim"],
        actions_standalone_hist_data, simple_env_config_vis["dt_sim_for_plot"]
    )

    print("\n모든 시각화 및 평가 시도 완료. 창을 닫으려면 모든 플롯 창을 닫아주세요.")
    plt.show() # 모든 plt.show(block=False) 이후 최종적으로 이벤트 루프를 유지