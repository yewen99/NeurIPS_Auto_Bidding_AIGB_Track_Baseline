import numpy as np
import math
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
import pandas as pd
# import datatable as dt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def run_test(file_path='./data/traffic/period-7.csv', model_name=None, model_param={}):
    """
    offline evaluation
    """

    data_loader = TestDataLoader(file_path=file_path)


    env = OfflineEnv()
    keys, test_dict = data_loader.keys, data_loader.test_dict
    overall_score = 0.0
    for key in keys:
        num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts, budget, cpa, category = data_loader.mock_data(key)
        agent = PlayerBiddingStrategy(model_name=model_name, model_param=model_param, budget=budget, cpa=cpa, category=category)
        print(agent.name)
        rewards = np.zeros(num_timeStepIndex)
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': []
        }

        for timeStep_index in range(num_timeStepIndex):
            # logger.info(f'Timestep Index: {timeStep_index + 1} Begin')

            pValue = pValues[timeStep_index]
            pValueSigma = pValueSigmas[timeStep_index]
            leastWinningCost = leastWinningCosts[timeStep_index]

            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(pValue.shape[0])
            else:

                bid = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                    history["historyBids"],
                                    history["historyAuctionResult"], history["historyImpressionResult"],
                                    history["historyLeastWinningCost"])

            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                          leastWinningCost)

            # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                    replace=False)
                bid[dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                              leastWinningCost)
                over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            agent.remaining_budget -= np.sum(tick_cost)
            rewards[timeStep_index] = np.sum(tick_conversion)
            temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
            history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(leastWinningCost)
            temAuctionResult = np.array(
                [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
            history["historyAuctionResult"].append(temAuctionResult)
            temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
            history["historyImpressionResult"].append(temImpressionResult)
            # logger.info(f'Timestep Index: {timeStep_index + 1} End')
        all_reward = np.sum(rewards)
        all_cost = agent.budget - agent.remaining_budget
        cpa_real = all_cost / (all_reward + 1e-10)
        cpa_constraint = agent.cpa
        score = getScore_nips(all_reward, cpa_real, cpa_constraint)
        overall_score+=score
        logger.info(f'Total Reward: {all_reward}')
        logger.info(f'Total Cost: {all_cost}')
        logger.info(f'CPA-real: {cpa_real}')
        logger.info(f'CPA-constraint: {cpa_constraint}')
        logger.info(f'Score: {score}')
    overall_score = overall_score/len(keys)
    logger.info(f'Period Score: {overall_score}')
    return overall_score


if __name__ == '__main__':
    model_param = {"use_noisy_condition": False,
                   "n_timesteps": 30}
    model_name = '/home/yewen001/CODE/ks/aigb/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/main/saved_model/DDtest/diffuser_best_epoch_loss_lr_0.0001_bs_1024_tau_0.01_step30.pt'
    run_test(model_name=model_name,
            model_param=model_param)
