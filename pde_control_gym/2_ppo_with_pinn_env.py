import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- DEVICE: {DEVICE} ---")

class TimeControlPINN_EnvironmentModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_layers = 5
        neurons_per_layer = 128
        layers = [input_dim] + [neurons_per_layer] * hidden_layers + [3] # u,v,p
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )
        self.activation = torch.tanh

    def forward(self, xytc):
        for i, linear_layer in enumerate(self.linears[:-1]):
            xytc = self.activation(linear_layer(xytc))
        return self.linears[-1](xytc)


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
        self.pinn_model = TimeControlPINN_EnvironmentModel(input_dim=pinn_input_dim).to(DEVICE)
        try:
            self.pinn_model.load_state_dict(torch.load(pretrained_pinn_model_path, map_location=DEVICE))
            print(f"사전 학습된 PINN 모델 로드 완료: {pretrained_pinn_model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"사전 학습된 PINN 모델 파일을 찾을 수 없습니다: {pretrained_pinn_model_path}\n"
                                    "먼저 train_time_control_pinn.py를 실행하여 모델을 학습/저장하세요.")
        except Exception as e:
            raise RuntimeError(f"PINN 모델 로드 중 오류 발생 ({pretrained_pinn_model_path}): {e}\n"
                               "모델 아키텍처와 저장된 state_dict가 일치하는지 확인하세요.")
        self.pinn_model.eval()

        self.action_space = spaces.Box(
            low=env_config.get("action_low", -1.0),
            high=env_config.get("action_high", 1.0),
            shape=(self.num_control_params,),
            dtype=np.float32
        )

        self.nx_obs = env_config.get("nx_obs", 20)
        self.ny_obs = env_config.get("ny_obs", 20)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.nx_obs, self.ny_obs, 2),
            dtype=np.float32
        )

        self.domain_L_pinn = env_config.get("domain_L_pinn", 1.0)
        self.domain_D_pinn = env_config.get("domain_D_pinn", 1.0)
        x_coords = torch.linspace(self.domain_L_pinn * 0, self.domain_L_pinn * 1, self.nx_obs, device=DEVICE)
        y_coords = torch.linspace(self.domain_D_pinn * 0, self.domain_D_pinn * 1, self.ny_obs, device=DEVICE)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
        self.obs_spatial_coords_flat = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        self.current_time = 0.0
        self.current_control_action = np.zeros(self.num_control_params, dtype=np.float32)
        self.current_uv_field = np.zeros((self.nx_obs, self.ny_obs, 2), dtype=np.float32)

        self.desire_states_uv = env_config.get("desire_states_uv", None)
        self.dt_desire = env_config.get("dt_desire", 1e-3)
        if self.desire_states_uv is not None:
            self.total_target_steps = self.desire_states_uv.shape[0]
            if self.desire_states_uv.shape[1] != self.nx_obs or self.desire_states_uv.shape[2] != self.ny_obs:
                print(f"경고: desire_states 그리드 크기 ({self.desire_states_uv.shape[1]}x{self.desire_states_uv.shape[2]})와 "
                      f"관찰 그리드 크기 ({self.nx_obs}x{self.ny_obs})가 다릅니다.")
                print("보상 계산 시 정확한 비교를 위해 그리드 크기를 일치시키거나 보간 처리가 필요합니다.")

        self.initial_control_action = env_config.get("initial_control_action", np.zeros(self.num_control_params, dtype=np.float32))

    def _get_pinn_prediction(self, time_val, control_action_np):
        control_action_torch = torch.tensor(control_action_np, dtype=torch.float32, device=DEVICE)
        num_spatial_points = self.obs_spatial_coords_flat.shape[0]

        time_tensor = torch.full((num_spatial_points, 1), time_val, dtype=torch.float32, device=DEVICE)
        expanded_control_action = control_action_torch.unsqueeze(0).repeat(num_spatial_points, 1)
        
        pinn_input = torch.cat([self.obs_spatial_coords_flat, time_tensor, expanded_control_action], dim=1)

        with torch.no_grad():
            uvp_flat = self.pinn_model(pinn_input)
        
        uv_flat = uvp_flat[:, :2]
        uv_reshaped = uv_flat.reshape(self.nx_obs, self.ny_obs, 2)
        return uv_reshaped.cpu().numpy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0.0
        self.current_step_env = 0
        self.current_control_action = np.copy(self.initial_control_action)
        self.current_uv_field = self._get_pinn_prediction(self.current_time, self.current_control_action)
        return self.current_uv_field, {}

    def step(self, action_from_ppo):
        self.current_control_action = action_from_ppo
        self.current_time += self.dt
        self.current_step_env += 1
        
        self.current_uv_field = self._get_pinn_prediction(self.current_time, self.current_control_action)
        
        reward = 0.0
        if self.desire_states_uv is not None:
            target_time_index = min(int(round(self.current_time / self.dt_desire)), self.total_target_steps - 1)
            
            if target_time_index < self.total_target_steps:
                 current_target_uv = self.desire_states_uv[target_time_index, :, :, :]
                 if current_target_uv.shape[0] != self.nx_obs or current_target_uv.shape[1] != self.ny_obs:
                    
                     mse_error = np.mean(self.current_uv_field**2) 
                 else:
                    mse_error = np.mean((self.current_uv_field - current_target_uv)**2)
                 reward = -mse_error 
            else:
                reward = -100.0 
        else:
            reward = -np.sum(action_from_ppo**2)

        done = self.current_time >= self.T_total - 1e-5 
        truncated = self.current_step_env >= self.max_steps_env
        if truncated and not done :
            done = True 

        return self.current_uv_field, reward, done, truncated, {}

if __name__ == "__main__":
    PRETRAINED_PINN_MODEL_PATH = "time_control_pinn.pth"

    if not os.path.exists(PRETRAINED_PINN_MODEL_PATH):
        print(f"오류: 사전 학습된 PINN 모델({PRETRAINED_PINN_MODEL_PATH})을 찾을 수 없습니다.")
        print("먼저 train_time_control_pinn.py를 실행하여 모델을 학습/저장하세요.")
        exit()

    nx_target, ny_target = 20, 20
    try:
        u_target_npz = np.load('target.npz')['u']
        v_target_npz = np.load('target.npz')['v']
        desire_states_from_npz = np.stack([u_target_npz, v_target_npz], axis=-1)
        nx_target, ny_target = desire_states_from_npz.shape[1], desire_states_from_npz.shape[2]
        print(f"target.npz 로드 완료. 형상: {desire_states_from_npz.shape} (NT, Nx, Ny, Channels)")
    except FileNotFoundError:
        print("경고: 'target.npz' 파일을 찾을 수 없습니다. 목표 상태 기반 보상이 작동하지 않을 수 있습니다.")
        desire_states_from_npz = None
    except KeyError:
        print("경고: 'target.npz' 파일에 'u' 또는 'v' 키가 없습니다.")
        desire_states_from_npz = None

    env_config = {
        "T_sim": 0.2,
        "dt_sim": 1e-2,
        "num_control_params": 1,
        "action_low": -1.0,
        "action_high": 1.0,
        "nx_obs": nx_target,
        "ny_obs": ny_target,
        "domain_L_pinn": 1.0,
        "domain_D_pinn": 1.0,
        "desire_states_uv": desire_states_from_npz,
        "dt_desire": 1e-3,
        "initial_control_action": np.array([0.0], dtype=np.float32)
    }

    PPO_POLICY_FILE = "ppo_pinn_control_policy.zip"
    PPO_LOG_DIR = "./ppo_pinn_control_log/"
    PPO_TOTAL_TIMESTEPS = 200_000

    def make_env_monitor(rank, seed=0):
        def _init():
            env = PINNNavierStokesEnv(PRETRAINED_PINN_MODEL_PATH, env_config)
            log_file = os.path.join(PPO_LOG_DIR, f"monitor_{rank}") if PPO_LOG_DIR else None
            
            env = Monitor(env, filename=log_file, allow_early_resets=True)
           
            return env
       
        return _init

    vec_env = DummyVecEnv([make_env_monitor(0)])


    checkpoint_callback = CheckpointCallback(
        save_freq=max(PPO_TOTAL_TIMESTEPS // 20, 10000),
        save_path=PPO_LOG_DIR,
        name_prefix="ppo_pinn_model"
    )
    
   
    eval_log_dir = os.path.join(PPO_LOG_DIR, 'eval_monitor_logs/')
    os.makedirs(eval_log_dir, exist_ok=True)
    eval_env_fn = lambda: Monitor(PINNNavierStokesEnv(PRETRAINED_PINN_MODEL_PATH, env_config), 
                                  filename=os.path.join(eval_log_dir, "eval_monitor_0"))
    eval_env_sb3 = DummyVecEnv([eval_env_fn])

    eval_callback = EvalCallback(eval_env_sb3, best_model_save_path=os.path.join(PPO_LOG_DIR, 'best_model'),
                                 log_path=os.path.join(PPO_LOG_DIR, 'eval_results'), 
                                 eval_freq=max(PPO_TOTAL_TIMESTEPS // 10, 5000),
                                 deterministic=True, render=False)

    PPO_LEARNING_RATE = 3e-4 

    if os.path.exists(PPO_POLICY_FILE) and False:
        print(f"--- 저장된 PPO 정책 로드: {PPO_POLICY_FILE} ---")
        ppo_model = PPO.load(PPO_POLICY_FILE, env=vec_env, device=DEVICE,
                             custom_objects={"learning_rate": PPO_LEARNING_RATE, "n_steps": 2048})
    else:
        print(f"--- PPO 에이전트 학습 시작 (PINN 기반 제어) ---")
        ppo_model = PPO("MlpPolicy", vec_env, verbose=1, device=DEVICE,
                        n_steps=2048,
                        batch_size=64,
                        n_epochs=10,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2,
                        ent_coef=0.0,
                        vf_coef=0.5,
                        max_grad_norm=0.5,
                        learning_rate=PPO_LEARNING_RATE,
                        tensorboard_log=PPO_LOG_DIR)
        
        print(f"PPO 학습 ({PPO_TOTAL_TIMESTEPS} 타임스텝) 시작...")
        ppo_model.learn(total_timesteps=PPO_TOTAL_TIMESTEPS,
                        callback=[checkpoint_callback, eval_callback], 
                        progress_bar=True)
        ppo_model.save(PPO_POLICY_FILE)
        print(f"PPO 정책 저장 완료: {PPO_POLICY_FILE}")

    print("\n--- 최종 학습된 PPO 에이전트 평가 ---")
    
    
    num_eval_episodes_final = 5
    total_rewards_manual_eval = []

    for episode in range(num_eval_episodes_final):
        obs = vec_env.reset() 
        done = [False]
        episode_reward = 0
        step_count = 0
        max_episode_steps = int(env_config["T_sim"] / env_config["dt_sim"])
        
        for _ in range(max_episode_steps):
            action, _ = ppo_model.predict(obs, deterministic=True)
            
            obs, reward, done, info = vec_env.step(action) 
            episode_reward += reward[0] 
            step_count +=1
            if done[0]: 
                break
        print(f"수동 평가 에피소드 {episode+1} 완료: 총 보상={episode_reward:.4f}, 스텝 수={step_count}")
        total_rewards_manual_eval.append(episode_reward)
        
    if num_eval_episodes_final > 0:
        print(f"수동 평균 평가 보상 ({num_eval_episodes_final} 에피소드): {np.mean(total_rewards_manual_eval):.4f} +/- {np.std(total_rewards_manual_eval):.4f}")

    print("PPO 기반 PINN 제어 학습 완료.")