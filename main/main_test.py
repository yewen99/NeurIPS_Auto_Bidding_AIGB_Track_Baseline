# import numpy as np
# import torch
# import os
# import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from run.run_evaluate import run_test

# torch.manual_seed(1)
# np.random.seed(1)

# if __name__ == "__main__":
#     test_model_path= 'saved_model/DDtest/diffuserbest_epoch_loss.pt'   
#     for test_dataisx in [7,8,9,10,11,12,13]:
#         print(f'evaluating priod-{test_dataisx}.csv')
#         run_test(test_model_path=test_model_path)

import numpy as np
import pandas as pd
import torch
import os
import sys
import glob
import random
from config import model_param
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run.run_evaluate import run_test
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


seed = 3407
"""Sets all possible random seeds so results can be reproduced"""
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
# tf.set_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

model_param["device"]="cuda"
if __name__ == "__main__":
    csv_files = glob.glob(os.path.join('./data/traffic', '*.csv'))
    # pt_files = glob.glob(os.path.join('./saved_model/DTtest', '*.pt'))
    pt_files = glob.glob(os.path.join('./saved_model/DDtest', '*.pt'))
    pt_names = [i.split("\\")[-1][:-3] for i in pt_files]
    eval_result = []
    for i, pt_f in enumerate(pt_files):
        score = 0.0
        for file in csv_files:
            print(f'Evaluating {pt_f} in {file} ===>')
            score += run_test(file_path=file, model_name=pt_names[i]+".pt", model_param=model_param)
            # break
        score /= len(csv_files)
        eval_result.append([pt_names[i], score])
        print("Average score of {}: {}".format(pt_names[i]+".pt", score))
        eval_result_csv = pd.DataFrame(eval_result, columns=["file", "score"]).sort_values(by="file")
        eval_result_csv.to_csv("eval_result.csv", index=False)

