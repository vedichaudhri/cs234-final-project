import pandas as pd
from data_processing import process_data, get_truth_cat
from plot import plot_fig
import math
import numpy as np
from baseline import FixedBaseline, ClinicalBaseline
from lin_ucb import Env, LinUCB, ThompsonSampler, LinOracle, SupervisedBandit


features, feature_df, dosage = process_data('./data/warfarin.csv')
true_cat = get_truth_cat(dosage)
env = Env(features, true_cat, dosage)
clinical_baseline = ClinicalBaseline()
fixed_baseline = FixedBaseline()
lin_ucb = LinUCB(3, len(features[0]), 0.1)
lin_thompson = ThompsonSampler(3, len(features[0]), 0.01)
lin_oracle = LinOracle(3, features, true_cat)
supervised_bandit = SupervisedBandit(len(features[0]))

algo = {}
algo["clinical_baseline"] = clinical_baseline
algo["fixed_baseline"] = fixed_baseline
algo["lin_ucb"] = lin_ucb
algo["lin_thompson"] = lin_thompson
algo["lin_oracle"] = lin_oracle
algo["supervised_bandit"] = supervised_bandit
reward_list = {}
regret_list = {}
for i in algo:
	reward_list[i] = []
	if i != "lin_oracle":
		regret_list[i] = []


for j in range(25):
	print(j)
	env.shuffle()
	for it in algo:
		algo[it].reset()

	f, i = env.get_element()
	reward = {}
	regret = {}
	for it in algo:
		reward[it] = []
		regret[it] = []
	time = 1
	while i is not None:
		for it in algo:
			if it == "lasso":
				continue
			arm = algo[it].choose_arm(f)
			if it != "lin_oracle":
				regret[it].append(algo["lin_oracle"].calc_regret(f, arm))
			r = env.get_reward(i, arm)
			reward[it].append(r)
			if it=="supervised_bandit":
				true_val = env.get_true_dosage(i)
				algo["supervised_bandit"].update_param(f, arm, true_val)
			else:
				algo[it].update_param(f, arm, r)

		f, i = env.get_element()
	for it in algo:
		reward_list[it].append(reward[it])
		if it != "lin_oracle":
			regret_list[it].append(regret[it])

new_list = {}
new_list_regret = {}
true_regret = {}
for it in algo:
	reward_list[it] = np.array(reward_list[it])
	new_list[it] = -1*reward_list[it]
	new_list[it] = np.cumsum(new_list[it], axis=1) / np.arange(1, len(reward_list[it][0])+1)
	if it != "lin_oracle":
		regret_list[it] = np.array(regret_list[it])
		new_list_regret[it] = np.cumsum(regret_list[it], axis=1)
		true_regret[it] = np.cumsum(-1*reward_list[it], axis=1)

accuracy = {}
for it in algo:
	sum_elem = 0
	for run in reward_list[it]:
		sum_elem += -1 * run[math.ceil(0.8*len(reward_list[it])):].sum()
	num_elem = len(reward_list[it][0][math.ceil(0.8*len(reward_list[it])):]) * len(reward_list[it])
	accuracy[it] = sum_elem/num_elem 

print(accuracy)
plot_data = {}
plot_data["label"] = {}
plot_data["color"] = {}
col_list = ["r", "y", "b", "g", "k", "m"]
counter = 0
for i in algo:
	plot_data["label"][i] = i
	plot_data["color"][i] = col_list[counter]
	counter += 1

plot_fig(new_list, plot_data, "Number of samples", "Fraction of incorrect decisions", "Plot of incorrect decisions vs samples", "incorrect")
plot_fig(new_list_regret, plot_data, "Number of samples", "Cumulative regret (wrt Linear Oracle)", "Cumulative regret vs Samples wrt the Linear Oracle", "regret")
plot_fig(true_regret, plot_data, "Number of samples", "Cumulative regret (using actual rewards)", "Cumulative regret vs Samples wrt the True Reward", "regret")
