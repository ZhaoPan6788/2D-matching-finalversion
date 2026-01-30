import os
import sys
import numpy as np
from scipy.optimize import differential_evolution

# 假设这些是你项目中的基础工具库
# 确保 base.impedance_analysis 存在或已将相关函数(test_ref_coef, get_ref_coef)定义在当前脚本中
from base.read_matrix import MatrixDataReader, PlasmaParameters
from base.impedance_analysis import test_ref_coef, get_ref_coef
from base.import_json import import_json
from base.save_json_file import save_json_file
from base.common import run_command  # 假设你有一个运行命令的封装

# --- 全局配置常量 ---
is_first_run = True        # restart
start_match = 5
match_count = 8
match_cycle = 10
GAMMA_TOL = 0.05    # 反射系数收敛阈值 (Gamma < 0.05 意味着 VSWR < 1.1)
POP_SIZE = 15       # 差分进化种群大小
MAX_ITER = 30       # 每次匹配周期的最大优化迭代次数

# 优化边界 (单位: Farads) - 根据实际 RF 匹配网络设计调整
BOUNDS = [
    (10e-12, 2000e-12), # Cm1 (Shunt, 并联电容)
    (10e-12, 1000e-12)  # Cm2 (Series, 串联电容)
]

def match(task_paths):
    """
    耦合 PIC 模拟与外部电路阻抗匹配的主控制逻辑。
    """
    if len(task_paths) == 0:
        return

    # 1. 环境与路径初始化
    ex = task_paths[0]
    base_path = os.path.split(ex[-1])[0]
    config_path = os.path.join(base_path, 'config.json')
    
    # 2D 模拟的数据输出文件通常包含电压电流波形
    # 请根据您的 Fortran 代码确认输出文件名，这里假设为 'EC.txt' 或类似的 electrical data
    data_file = os.path.join(base_path, 'EC.txt') 

    # 读取配置文件
    d = import_json(config_path)

    # 2. 状态判定与初始化 (断点续算逻辑)
    # is_first_run, start_match, match_cycle, match_count 假设为外部定义的全局变量或参数
    if is_first_run:            
        # --- 冷启动 ---
        n_match = 0
        # clear_one(base_path) # 如有清理旧数据的函数，在此调用
        print(f"[Init] Starting fresh simulation at {base_path}")
    else:
        # --- 续算 ---
        nrun_c = d['control']['nrun']
        if nrun_c >= start_match:
            # 计算当前已完成多少个匹配周期
            n_match = (nrun_c - start_match) // match_cycle
            print(f"[Restart] Detected completed match cycles: {n_match}")
        else:
            n_match = 0
            # clear_one(base_path)
            print("[Restart] Resetting to pre-burn phase.")

    # 3. 主循环：推进时间 -> 分析波形 -> 优化电路
    while n_match <= match_count:
        print(f"\n>>> Processing Match Cycle: {n_match} / {match_count}")

        # ------------------------------------------------------
        # A. 设置模拟步数 (Time Stepping Control)
        # ------------------------------------------------------
        if n_match == 0:
            # 阶段 0: 预燃 (Pre-burn)
            # 仅运行到 start_match，不进行匹配，等待等离子体建立准稳态
            target_nrun = start_match
            print(f"[Phase 0] Pre-burn plasma to step {target_nrun}...")
        else:
            # 阶段 N: 匹配与演化
            # 在现有基础上增加一个周期
            target_nrun = start_match + match_cycle * n_match
            print(f"[Phase {n_match}] Advancing simulation to step {target_nrun}...")

        # 更新 config 并保存
        d['control']['nrun'] = target_nrun
        # 开启重启标志 (除了第0次可能是冷启动)
        if 'restart' in d['control']:
            d['control']['restart'] = (target_nrun > start_match) or (not is_first_run)
            
        save_json_file(base_path, 'config.json', d)

        # ------------------------------------------------------
        # B. 执行 PIC 核心 (Run PIC Core)
        # ------------------------------------------------------
        # 这里的 run_pic 是您的运行脚本，如 mpirun ...
        # 确保它会阻塞直到 Fortran 程序结束
        run_simulation_cmd(base_path) 

        # ------------------------------------------------------
        # C. 阻抗分析与优化 (Impedance Matching & Optimization)
        # ------------------------------------------------------
        # 只有在预燃结束后 (n_match > 0) 且数据文件存在时才进行匹配
        if n_match > 0:
            if not os.path.exists(data_file):
                print(f"[Error] Data file {data_file} not found after simulation!")
                break

            # C1. 读取电压电流数据
            md = MatrixDataReader([data_file])
            md.read_matrices()
            
            # C2. 计算等离子体复阻抗 Z_load
            # PlasmaParameters 内部应包含 FFT 分析提取基波分量的逻辑
            c = PlasmaParameters(md, d)
            
            if not hasattr(c, 'Z_load'):
                print("[Error] Failed to calculate Z_load.")
                break

            # C3. 读取当前电容值 (兼容 1D/2D 结构)
            try:
                if 'imn' in d['circuit']: # 2D 结构常见
                    curr_Cm1 = d['circuit']['imn']['Cm1']
                    curr_Cm2 = d['circuit']['imn']['Cm2']
                else: # 1D 结构或旧版
                    curr_Cm1 = d['circuit']['circuits_list'][0]['Cm1']
                    curr_Cm2 = d['circuit']['circuits_list'][0]['Cm2']
            except KeyError:
                print("[Error] Cannot find Cm1/Cm2 in config.json")
                break

            # C4. 评估当前反射系数 Gamma
            g_curr = test_ref_coef([curr_Cm1, curr_Cm2], c)
            print(f"   [Analysis] Z_load = {c.Z_load:.2f} Ohm")
            print(f"   [Current]  Cm1={curr_Cm1:.2e}, Cm2={curr_Cm2:.2e} => Gamma={g_curr:.4f}")

            # C5. 判断收敛
            if g_curr < GAMMA_TOL:
                print(f"   [Converged] Gamma {g_curr:.4f} < {GAMMA_TOL}. Keeping parameters.")
                # 即使收敛，我们通常也继续跑下一个周期以观察等离子体漂移，
                # 或者您可以选择在此 break 结束整个匹配任务。
            else:
                # C6. 执行差分进化优化
                print("   [Optimization] Tuning Matching Network...")
                result = differential_evolution(
                    get_ref_coef, 
                    BOUNDS, 
                    strategy='best1bin',
                    maxiter=MAX_ITER, 
                    popsize=POP_SIZE, 
                    tol=1e-2, 
                    mutation=(0.5, 1.0), 
                    recombination=0.7,
                    args=[c] # 将当前等离子体状态作为参数传递
                )
                
                best_Cm1, best_Cm2 = result.x
                g_pred = test_ref_coef(result.x, c)
                print(f"   [New Best] Cm1={best_Cm1:.2e}, Cm2={best_Cm2:.2e} => Predicted Gamma={g_pred:.4f}")

                # C7. 将优化结果写回 Config
                if 'imn' in d['circuit']:
                    d['circuit']['imn']['Cm1'] = best_Cm1
                    d['circuit']['imn']['Cm2'] = best_Cm2
                else:
                    d['circuit']['circuits_list'][0]['Cm1'] = best_Cm1
                    d['circuit']['circuits_list'][0]['Cm2'] = best_Cm2
                
                # 立即保存，以便下一轮循环（或手动检查）生效
                save_json_file(base_path, 'config.json', d)

        # ------------------------------------------------------
        # D. 推进循环
        # ------------------------------------------------------
        n_match += 1

def run_simulation_cmd(cwd):
    """封装原本的 os.system 调用，建议根据环境修改"""
    # 示例: mpirun -np 4 ./bin/iPM2D
    # 实际项目中，cmd 应该从外部传入或配置中读取
    run_cmd = 'mpirun -np 4 ./pic2d-mpi' 
    print(f"Executing: {run_cmd} in {cwd}")
    ret = os.system(f"cd {cwd} && {run_cmd} > run.log 2>&1")
    if ret != 0:
        print(f"[Warning] Simulation returned non-zero exit code: {ret}")