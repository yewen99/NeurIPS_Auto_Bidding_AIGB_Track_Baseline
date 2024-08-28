import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from run.run_decision_diffuser import run_decision_diffuser

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    # bss = [64, 256, 512, 1024]
    # taus = [1e-3, 1e-2, 1e-1]
    # lrs = [1e-5, 1e-4, 1e-3]
    # for bs in bss:
    batch_size = 1024
    lr = 1e-3
    tau = 0.01
    time_steps = 100
    print(f'training with batch_size:{batch_size} lr:{lr} tau:{tau} time_steps:{time_steps}')
    run_decision_diffuser(train_epoch=1000,
        batch_size=batch_size,
        gamma=1, 
        tau=tau, 
        lr=lr,
        n_timesteps=time_steps,
        )
