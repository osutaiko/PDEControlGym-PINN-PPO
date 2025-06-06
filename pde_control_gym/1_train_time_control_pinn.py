import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

# --- 기본 환경 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- PINN 학습 장치: {DEVICE} ---")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- 물리/격자 상수 및 시뮬레이션 파라미터 ---
RHO = 1.0
MU = 1.0
U_IN_DEFAULT = 1.0

DOMAIN_X_MIN, DOMAIN_X_MAX = 0.0, 1.0
DOMAIN_Y_MIN, DOMAIN_Y_MAX = 0.0, 1.0
T_MAX = 0.2

# PINN 학습 파라미터
LEARNING_RATE_ADAM = 1e-3
LEARNING_RATE_LBFGS = 1.0
NUM_EPOCHS_ADAM = 15000
NUM_LBFGS_STEPS = 5000
LOG_FREQUENCY = 1000
LBFGS_LOG_FREQUENCY = 100 # L-BFGS는 더 자주 로깅할 수 있음 (스텝 기준)

# 콜로케이션 포인트 수
N_DOMAIN = 2500
N_BOUNDARY = 500
N_INITIAL = 500

# 제어 파라미터 개수 및 범위
NUM_CONTROL_PARAMS = 1
CONTROL_PARAM_MIN = -0.5
CONTROL_PARAM_MAX = 0.5

MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "time_control_pinn.pth")
PINN_TRAINING_LOG_SAVE_PATH = os.path.join(ROOT_DIR, "pinn_training_log.csv") 

# --- 시간 및 제어 인지 PINN 모델 정의 ---
class TimeControlPINN_Trainer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_layers = 5
        neurons_per_layer = 128
        layers = [input_dim] + [neurons_per_layer] * hidden_layers + [3] # u,v,p
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )
        self.activation = torch.tanh

    def forward(self, xytc): # xytc: (N, 4) 텐서 [x, y, t, control_val]
        for i, linear_layer in enumerate(self.linears[:-1]):
            xytc = self.activation(linear_layer(xytc))
        return self.linears[-1](xytc)

# --- 유틸리티 함수: 콜로케이션 포인트 생성 ---
def generate_training_data(device):
    x_domain = torch.rand(N_DOMAIN, 1, device=device) * (DOMAIN_X_MAX - DOMAIN_X_MIN) + DOMAIN_X_MIN
    y_domain = torch.rand(N_DOMAIN, 1, device=device) * (DOMAIN_Y_MAX - DOMAIN_Y_MIN) + DOMAIN_Y_MIN
    t_domain = torch.rand(N_DOMAIN, 1, device=device) * T_MAX
    c_domain = torch.rand(N_DOMAIN, NUM_CONTROL_PARAMS, device=device) * (CONTROL_PARAM_MAX - CONTROL_PARAM_MIN) + CONTROL_PARAM_MIN
    domain_points = torch.cat([x_domain, y_domain, t_domain, c_domain], dim=1).requires_grad_(True)

    y_inlet = torch.rand(N_BOUNDARY, 1, device=device) * (DOMAIN_Y_MAX - DOMAIN_Y_MIN) + DOMAIN_Y_MIN
    t_inlet = torch.rand(N_BOUNDARY, 1, device=device) * T_MAX
    c_inlet = torch.rand(N_BOUNDARY, NUM_CONTROL_PARAMS, device=device) * (CONTROL_PARAM_MAX - CONTROL_PARAM_MIN) + CONTROL_PARAM_MIN
    inlet_points = torch.cat([torch.full_like(y_inlet, DOMAIN_X_MIN), y_inlet, t_inlet, c_inlet], dim=1).requires_grad_(True)

    y_outlet = torch.rand(N_BOUNDARY, 1, device=device) * (DOMAIN_Y_MAX - DOMAIN_Y_MIN) + DOMAIN_Y_MIN
    t_outlet = torch.rand(N_BOUNDARY, 1, device=device) * T_MAX
    c_outlet = torch.rand(N_BOUNDARY, NUM_CONTROL_PARAMS, device=device) * (CONTROL_PARAM_MAX - CONTROL_PARAM_MIN) + CONTROL_PARAM_MIN
    outlet_points = torch.cat([torch.full_like(y_outlet, DOMAIN_X_MAX), y_outlet, t_outlet, c_outlet], dim=1).requires_grad_(True)

    x_wall_b = torch.rand(N_BOUNDARY, 1, device=device) * (DOMAIN_X_MAX - DOMAIN_X_MIN) + DOMAIN_X_MIN
    t_wall_b = torch.rand(N_BOUNDARY, 1, device=device) * T_MAX
    c_wall_b = torch.rand(N_BOUNDARY, NUM_CONTROL_PARAMS, device=device) * (CONTROL_PARAM_MAX - CONTROL_PARAM_MIN) + CONTROL_PARAM_MIN
    wall_bottom_points = torch.cat([x_wall_b, torch.full_like(x_wall_b, DOMAIN_Y_MIN), t_wall_b, c_wall_b], dim=1).requires_grad_(True)

    x_wall_t = torch.rand(N_BOUNDARY, 1, device=device) * (DOMAIN_X_MAX - DOMAIN_X_MIN) + DOMAIN_X_MIN
    t_wall_t = torch.rand(N_BOUNDARY, 1, device=device) * T_MAX
    c_wall_t = torch.rand(N_BOUNDARY, NUM_CONTROL_PARAMS, device=device) * (CONTROL_PARAM_MAX - CONTROL_PARAM_MIN) + CONTROL_PARAM_MIN
    wall_top_points = torch.cat([x_wall_t, torch.full_like(x_wall_t, DOMAIN_Y_MAX), t_wall_t, c_wall_t], dim=1).requires_grad_(True)

    x_initial = torch.rand(N_INITIAL, 1, device=device) * (DOMAIN_X_MAX - DOMAIN_X_MIN) + DOMAIN_X_MIN
    y_initial = torch.rand(N_INITIAL, 1, device=device) * (DOMAIN_Y_MAX - DOMAIN_Y_MIN) + DOMAIN_Y_MIN
    c_initial = torch.rand(N_INITIAL, NUM_CONTROL_PARAMS, device=device) * (CONTROL_PARAM_MAX - CONTROL_PARAM_MIN) + CONTROL_PARAM_MIN
    initial_points = torch.cat([x_initial, y_initial, torch.zeros_like(x_initial), c_initial], dim=1).requires_grad_(True)

    return domain_points, inlet_points, outlet_points, wall_bottom_points, wall_top_points, initial_points

# --- PDE 잔차 및 손실 함수 ---
def get_pde_residuals_and_losses(model, domain_points_with_grad):
    xytc = domain_points_with_grad

    uvp = model(xytc)
    u, v, p = uvp[:,0:1], uvp[:,1:2], uvp[:,2:3]
    
    grad_u_outputs = torch.autograd.grad(u.sum(), xytc, create_graph=True, retain_graph=True)[0]
    du_dx = grad_u_outputs[:, 0:1]
    du_dy = grad_u_outputs[:, 1:2]
    du_dt = grad_u_outputs[:, 2:3]

    grad_v_outputs = torch.autograd.grad(v.sum(), xytc, create_graph=True, retain_graph=True)[0]
    dv_dx = grad_v_outputs[:, 0:1]
    dv_dy = grad_v_outputs[:, 1:2]
    dv_dt = grad_v_outputs[:, 2:3]

    grad_p_outputs = torch.autograd.grad(p.sum(), xytc, create_graph=True, retain_graph=True)[0]
    dp_dx = grad_p_outputs[:, 0:1]
    dp_dy = grad_p_outputs[:, 1:2]

    du_dxx = torch.autograd.grad(du_dx.sum(), xytc, create_graph=True, retain_graph=True)[0][:, 0:1]
    du_dyy = torch.autograd.grad(du_dy.sum(), xytc, create_graph=True, retain_graph=True)[0][:, 1:2]
    dv_dxx = torch.autograd.grad(dv_dx.sum(), xytc, create_graph=True, retain_graph=True)[0][:, 0:1]
    dv_dyy = torch.autograd.grad(dv_dy.sum(), xytc, create_graph=True, retain_graph=True)[0][:, 1:2]

    continuity_eq = du_dx + dv_dy
    momentum_x_eq = RHO * (du_dt + u * du_dx + v * du_dy) + dp_dx - MU * (du_dxx + du_dyy)
    momentum_y_eq = RHO * (dv_dt + u * dv_dx + v * dv_dy) + dp_dy - MU * (dv_dxx + dv_dyy)

    loss_pde_c = torch.mean(continuity_eq**2)
    loss_pde_mx = torch.mean(momentum_x_eq**2)
    loss_pde_my = torch.mean(momentum_y_eq**2)
    
    total_loss_pde = loss_pde_c + loss_pde_mx + loss_pde_my
    return total_loss_pde, loss_pde_c, loss_pde_mx, loss_pde_my # <<<< 개별 PDE 손실도 반환

# --- 경계 조건 및 초기 조건 손실 ---
def get_boundary_initial_losses(model, inlet_pts, outlet_pts, wall_b_pts, wall_t_pts, initial_pts):
    uvp_inlet = model(inlet_pts)
    u_inlet, v_inlet = uvp_inlet[:,0:1], uvp_inlet[:,1:2]
    loss_bc_inlet_u = torch.mean((u_inlet - U_IN_DEFAULT)**2)
    loss_bc_inlet_v = torch.mean((v_inlet - 0.0)**2)
    loss_bc_inlet = loss_bc_inlet_u + loss_bc_inlet_v

    uvp_outlet = model(outlet_pts)
    p_outlet = uvp_outlet[:,2:3]
    loss_bc_outlet = torch.mean(p_outlet**2)

    uvp_wall_b = model(wall_b_pts)
    u_wall_b, v_wall_b = uvp_wall_b[:,0:1], uvp_wall_b[:,1:2]
    loss_bc_wall_b = torch.mean(u_wall_b**2) + torch.mean(v_wall_b**2)

    control_val_top_wall = wall_t_pts[:, 3:4] 
    uvp_wall_t = model(wall_t_pts)
    u_wall_t, v_wall_t = uvp_wall_t[:,0:1], uvp_wall_t[:,1:2]
    loss_bc_wall_t_u = torch.mean((u_wall_t - control_val_top_wall)**2)
    loss_bc_wall_t_v = torch.mean(v_wall_t**2)
    loss_bc_wall_t = loss_bc_wall_t_u + loss_bc_wall_t_v

    total_loss_bc = loss_bc_inlet + loss_bc_outlet + loss_bc_wall_b + loss_bc_wall_t

    uvp_initial = model(initial_pts)
    u_initial, v_initial, p_initial = uvp_initial[:,0:1], uvp_initial[:,1:2], uvp_initial[:,2:3]
    loss_ic_u = torch.mean(u_initial**2)
    loss_ic_v = torch.mean(v_initial**2)
    loss_ic_p = torch.mean(p_initial**2) 
    total_loss_ic = loss_ic_u + loss_ic_v + loss_ic_p

    return total_loss_bc, total_loss_ic

# --- 학습 루프 ---
if __name__ == "__main__":
    pinn_model_trainer = TimeControlPINN_Trainer(input_dim=2+1+NUM_CONTROL_PARAMS).to(DEVICE)
    
    print("학습 데이터 생성 중...")
    domain_pts, inlet_pts, outlet_pts, wall_b_pts, wall_t_pts, initial_pts = generate_training_data(DEVICE)
    print("학습 데이터 생성 완료.")

    pinn_training_logs = []

    # --- 1단계: Adam 옵티마이저 사용 ---
    print(f"PINN 학습 시작 (Adam, 총 {NUM_EPOCHS_ADAM} 에폭)...")
    optimizer_adam = optim.Adam(pinn_model_trainer.parameters(), lr=LEARNING_RATE_ADAM)
    
    for epoch in tqdm(range(NUM_EPOCHS_ADAM), desc="Adam Training"):
        optimizer_adam.zero_grad()

        loss_pde_total, loss_pde_c, loss_pde_mx, loss_pde_my = get_pde_residuals_and_losses(pinn_model_trainer, domain_pts)
        loss_bc, loss_ic = get_boundary_initial_losses(pinn_model_trainer, inlet_pts, outlet_pts,
                                                       wall_b_pts, wall_t_pts, initial_pts)
        
        total_loss = loss_pde_total + loss_bc + loss_ic
        
        total_loss.backward()
        optimizer_adam.step()

        if (epoch + 1) % LOG_FREQUENCY == 0:
            log_entry = {
                'optimizer': 'Adam',
                'epoch_or_step': epoch + 1,
                'total_loss': total_loss.item(),
                'pde_loss': loss_pde_total.item(),
                'pde_loss_continuity': loss_pde_c.item(),
                'pde_loss_momentum_x': loss_pde_mx.item(),
                'pde_loss_momentum_y': loss_pde_my.item(),# 개별 PDE 항 로깅
                'bc_loss': loss_bc.item(),
                'ic_loss': loss_ic.item()
            }
            pinn_training_logs.append(log_entry)
            tqdm.write(f"Adam Epoch [{epoch+1}/{NUM_EPOCHS_ADAM}], Total Loss: {total_loss.item():.4e}, "
                       f"PDE: {loss_pde_total.item():.3e}, BC: {loss_bc.item():.3e}, IC: {loss_ic.item():.3e}")

    print("Adam 학습 완료.")
    current_loss_after_adam = total_loss.item() # 마지막 Adam 손실

    # --- 2단계: L-BFGS 옵티마이저 사용 ---
    print(f"\nPINN 학습 시작 (L-BFGS, 최대 {NUM_LBFGS_STEPS} 스텝)...")
    optimizer_lbfgs = optim.LBFGS(
        pinn_model_trainer.parameters(),
        lr=LEARNING_RATE_LBFGS,
        max_iter=20,
        max_eval=None,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    loss_tracker_lbfgs = {'current_loss': float('inf'), 'pde': float('inf'), 'bc': float('inf'), 'ic': float('inf'),
                          'pde_c': float('inf'), 'pde_mx': float('inf'), 'pde_my': float('inf')}


    def closure():
        optimizer_lbfgs.zero_grad()
        loss_pde_lbfgs_total, loss_pde_c_l, loss_pde_mx_l, loss_pde_my_l = get_pde_residuals_and_losses(pinn_model_trainer, domain_pts)
        loss_bc_lbfgs, loss_ic_lbfgs = get_boundary_initial_losses(pinn_model_trainer, inlet_pts, outlet_pts,
                                                                  wall_b_pts, wall_t_pts, initial_pts)
        total_loss_lbfgs = loss_pde_lbfgs_total + loss_bc_lbfgs + loss_ic_lbfgs
        total_loss_lbfgs.backward()
        
        loss_tracker_lbfgs['current_loss'] = total_loss_lbfgs.item()
        loss_tracker_lbfgs['pde'] = loss_pde_lbfgs_total.item()
        loss_tracker_lbfgs['pde_c'] = loss_pde_c_l.item()
        loss_tracker_lbfgs['pde_mx'] = loss_pde_mx_l.item()
        loss_tracker_lbfgs['pde_my'] = loss_pde_my_l.item()
        loss_tracker_lbfgs['bc'] = loss_bc_lbfgs.item()
        loss_tracker_lbfgs['ic'] = loss_ic_lbfgs.item()
        return total_loss_lbfgs

    for i in tqdm(range(NUM_LBFGS_STEPS), desc="L-BFGS Training"):
        prev_loss = loss_tracker_lbfgs['current_loss']
        optimizer_lbfgs.step(closure) # closure 내부에서 loss_tracker_lbfgs가 업데이트됨
        
        if (i + 1) % LBFGS_LOG_FREQUENCY == 0:
            log_entry = {
                'optimizer': 'LBFGS',
                'epoch_or_step': i + 1,
                'total_loss': loss_tracker_lbfgs['current_loss'],
                'pde_loss': loss_tracker_lbfgs['pde'],
                'pde_loss_continuity': loss_tracker_lbfgs['pde_c'],
                'pde_loss_momentum_x': loss_tracker_lbfgs['pde_mx'],
                'pde_loss_momentum_y': loss_tracker_lbfgs['pde_my'],
                'bc_loss': loss_tracker_lbfgs['bc'],
                'ic_loss': loss_tracker_lbfgs['ic']
            }
            pinn_training_logs.append(log_entry)
            tqdm.write(f"L-BFGS Step [{i+1}/{NUM_LBFGS_STEPS}], Total Loss: {loss_tracker_lbfgs['current_loss']:.4e}, "
                       f"PDE: {loss_tracker_lbfgs['pde']:.3e}, BC: {loss_tracker_lbfgs['bc']:.3e}, IC: {loss_tracker_lbfgs['ic']:.3e}")
        
        if abs(prev_loss - loss_tracker_lbfgs['current_loss']) < 1e-8 and i > 10:
            tqdm.write(f"L-BFGS 수렴됨 (Step {i+1}).")
            break
        if loss_tracker_lbfgs['current_loss'] > current_loss_after_adam * 2.0 and i > 50 :
            tqdm.write(f"L-BFGS 손실 발산 가능성으로 중단 (Step {i+1}). Loss: {loss_tracker_lbfgs['current_loss']:.4e}")
            break
        if np.isnan(loss_tracker_lbfgs['current_loss']):
            tqdm.write(f"L-BFGS 손실이 NaN이 되어 중단 (Step {i+1}).")
            break

    print("L-BFGS 학습 완료.")
    torch.save(pinn_model_trainer.state_dict(), MODEL_SAVE_PATH)
    print(f"학습된 PINN 모델 저장 완료: {MODEL_SAVE_PATH}")

    # --- <<<< 학습 로그 CSV 파일로 저장 >>>> ---
    if pinn_training_logs:
        df_pinn_log = pd.DataFrame(pinn_training_logs)
        df_pinn_log.to_csv(PINN_TRAINING_LOG_SAVE_PATH, index=False)
        print(f"PINN 학습 로그 저장 완료: {PINN_TRAINING_LOG_SAVE_PATH}")
    else:
        print("저장할 PINN 학습 로그 데이터가 없습니다.")

    # --- 간단한 테스트 ---
    print("\n--- 학습된 모델 테스트 (예시) ---")
    test_x = torch.tensor([[0.5]], device=DEVICE)
    test_y = torch.tensor([[0.5]], device=DEVICE)
    test_t = torch.tensor([[T_MAX / 2]], device=DEVICE)
    test_control = torch.tensor([[0.1]], device=DEVICE)
    test_input = torch.cat([test_x, test_y, test_t, test_control], dim=1)
    
    pinn_model_trainer.eval()
    with torch.no_grad():
        test_output_uvp = pinn_model_trainer(test_input)
    print(f"테스트 입력: x={test_x.item()}, y={test_y.item()}, t={test_t.item()}, control={test_control.item()}")
    print(f"테스트 출력 (u,v,p): {test_output_uvp.cpu().numpy()}")