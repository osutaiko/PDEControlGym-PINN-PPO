## About
[PDE Control Gym](https://github.com/lukebhan/PDEControlGym) is the first open benchmark that lets model-free RL control representative Partial Differential Equations (PDEs) from their boundaries, such as 2D Navier-stokes flow. While Proximal Policy Optimization (PPO) does work, it requires millions of steps and still shows unstable, oscillating behavior. We found that this is because the reward does not reflect the physical property of the PDE. 

In **PINN-PPO**, we add a lightweight Physics-Informed Neural Network (PINN) residual term to the reward, keeping the agent model-free yet physics-aware. This provides real-time PDE-error feedback, guiding the agent towards faster convergence with improved stability.

## Getting Started
1. `pip install -e .` : Install dependencies
2. `cd pde_control_gym` : Navigate to the working directory (PDEControlGym-main)
3. Train the models:
   - `python 0_ppo_simple_env_trainer.py` : Train the standalone PPO agent
   - `python 1_train_time_control_pinn.py` : Train the PINN agent for time-dependent control
   - `python 2_ppo_with_pinn_env.py` : Train the PINN-PPO agent
   - `python 3_compare_pinn_vs_standalone_ppo.py` : Evaluate and compare the trained PINN-PPO and standalone PPO agents
     
   After training, multiple windows with diagrams will pop up (the same will also be saved in the `final_comparison_log` folder). 

## Contributors
- Hyojin Kim
- Chankyu Lee
- Minjoon Jeong
- Eungi Hong
- Dongheon Han

## [Original Repository](https://github.com/lukebhan/PDEControlGym)

<a href="#"><img alt="PDE ContRoL Gym" src="PDEGymLogo.png" width="100%"/></a>

<p>
<a href='https://pdecontrolgym.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/pdecontrolgym/badge/?version=latest' alt='Documentation Status' />
    </a>
<a href=https://arxiv.org/abs/2302.14265> 
    <img src="https://img.shields.io/badge/arXiv-2302.14265-008000.svg" alt="arXiv Status" />
</a>
</p>
