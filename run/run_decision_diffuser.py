import torch
from bidding_train_env.baseline.dd.DFUSER import (DFUSER)
import time
from bidding_train_env.baseline.dd.dataset import aigb_dataset
from torch.utils.data import DataLoader
import tqdm


def run_decision_diffuser(
        save_path="saved_model/DDtest",
        train_epoch=1,
        batch_size=1000,
        gamma=1, 
        tau=0.01, 
        lr=1e-4,
        network_random_seed=200):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("train_epoch", train_epoch)
    print("batch-size", batch_size)

    algorithm = DFUSER(gamma=gamma, tau=tau, lr=lr,
                 network_random_seed=network_random_seed,
                 )
    algorithm = algorithm.to(device)

    args_dict = {'data_version': 'monk_data_small'}
    dataset = aigb_dataset(algorithm.step_len, **args_dict)
    dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, num_workers=2, pin_memory=True)

    # 参数数量
    total_params = sum(p.numel() for p in algorithm.parameters())
    print(f"参数数量：{total_params}")

    # 3. 迭代训练

    epi = 1
    best_score = 1e10
    
    for epoch in tqdm.tqdm(range(0, train_epoch), desc='training decision diffusion...'):

        record_epoch_loss = 0.
        record_epoch_diff_loss = 0.
        record_epoch_inv_loss = 0.
        

        for batch_index, (states, actions, returns, masks) in enumerate(dataloader):
            states.to(device)
            actions.to(device)
            returns.to(device)
            masks.to(device)

            start_time = time.time()

            # 训练
            all_loss, (diffuse_loss, inv_loss) = algorithm.trainStep(states, actions, returns, masks)
            all_loss = all_loss.detach().clone()
            diffuse_loss = diffuse_loss.detach().clone()
            inv_loss = inv_loss.detach().clone()
            end_time = time.time()
            # print(
            #     f"epoch {epoch}/{train_epoch} ==> batch: 第{epi}个batch训练时间为: {end_time - start_time} s, all_loss: {all_loss}, diffuse_loss: {diffuse_loss}, inv_loss: {inv_loss}")
            
            record_epoch_loss += all_loss
            record_epoch_diff_loss += diffuse_loss
            record_epoch_inv_loss += inv_loss
            epi += 1

        record_epoch_loss /= len(dataloader)
        record_epoch_diff_loss /= len(dataloader)
        record_epoch_inv_loss /= len(dataloader)
        print(f'epoch: {epoch}/{train_epoch} ==> epoch_loss:{record_epoch_loss} epoch_diff_loss:{record_epoch_diff_loss} epoch_inv_loss:{record_epoch_inv_loss}')
        if record_epoch_loss < best_score:
            best_score = record_epoch_loss
            algorithm.save_net(save_path, save_name=f'_best_epoch_loss_lr_{lr}_bs_{batch_size}_tau_{tau}')
            # algorithm.save_net(save_path)
            print(f'saved at epoch {epoch} with best epoch all loss: {best_score}!')
        
        # if record_epoch_inv_loss < best_score:
        #     algorithm.save_net(save_path, save_name=f'best_epoch_inv_loss')

    
    # algorithm.save_net(save_path, epi)


if __name__ == '__main__':
    run_decision_diffuser()
