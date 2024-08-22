# model_param = {
#     "step_num": 1000000,
#     "save_step": 50000,
#     "dir": "./data/trajectory/trajectory_data_all.csv",
#     "hidden_size": 512,
#     "learning_rate": 0.0001,
#     "time_dim": 8,
#     "batch_size": 128,
#     "device": "cuda:0",
#     "block_config": {
#         "n_ctx": 1024,
#         "n_embd": 512,
#         "n_layer": 6,
#         "n_head": 8,
#         "n_inner": 512,
#         "activation_function": "relu",
#         "n_position": 1024,
#         "resid_pdrop": 0.1,
#         "attn_pdrop": 0.1
#     }
# }

model_param = {
    "step_num": 1000000,
    "save_step": 30000,
    "dir": "./data/trajectory/trajectory_data_all.csv",
    "hidden_size": 512,
    "learning_rate": 0.0001,
    "time_dim": 8,
    "batch_size": 128,
    "device": "cuda:0",
    "block_config": {
        "n_ctx": 1024,
        "n_embd": 512,
        "n_layer": 8,
        "n_head": 16,
        "n_inner": 1024,
        "activation_function": "relu",
        "n_position": 1024,
        "resid_pdrop": 0.1,
        "attn_pdrop": 0.1
    }
}