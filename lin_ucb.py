import numpy as np
import pandas as pd
from data_processing import process_data, get_truth_cat
from plot import plot_fig
import math
from sklearn import linear_model

class Env:
	def __init__(self, data, true_categories, true_val):
		self.data = data
		self.true_categories = true_categories
		self.tru_val = true_val
		self.num_data = len(data)
		self.counter = 0

	def shuffle(self):
		p = np.random.permutation(self.num_data)
		self.data = self.data[p]
		self.true_categories = self.true_categories[p]
		self.tru_val = self.tru_val[p]
		self.counter = 0

	def get_element(self):
		if self.counter == self.num_data:
			return None, None
		self.counter += 1
		return self.data[self.counter-1].reshape((-1,1)), self.counter-1

	def get_reward(self, index, pred):
		if self.true_categories[index] == pred:
			return 0
		return -1

	def get_true_dosage(self, index):
		return self.tru_val[index]

class LinUCB:
	def __init__(self, num_arms, num_features, alpha):
		self.num_arms = num_arms
		self.A = [np.eye(num_features) for i in range(num_arms)]
		self.B = [np.zeros(num_features).reshape((-1,1)) for i in range(num_arms)]
		self.A_inv = [np.eye(num_features) for i in range(num_arms)]
		self.theta_hat = [np.zeros(num_features).reshape((-1,1)) for i in range(num_arms)]
		self.num_features = num_features
		self.alpha = alpha

	def choose_arm(self, x):
		reward = []
		for a in range(self.num_arms):
			p = np.matmul(np.transpose(self.theta_hat[a]),x)[0][0] 
			+ self.alpha * np.sqrt(np.matmul(np.matmul(np.transpose(x), self.A_inv[a]), x))[0][0]
			reward.append(p)
		reward = np.array(reward)
		return reward.argmax()

	def update_param(self, x, arm, reward):
		self.A[arm] += np.matmul(x, np.transpose(x))
		self.B[arm] += reward*x
		self.A_inv[arm] = np.linalg.inv(self.A[arm])
		self.theta_hat[arm] = np.matmul(self.A_inv[arm], self.B[arm])

	def reset(self):
		self.A = [np.eye(self.num_features) for i in range(self.num_arms)]
		self.B = [np.zeros(self.num_features).reshape((-1,1)) for i in range(self.num_arms)]
		self.A_inv = [np.eye(self.num_features) for i in range(self.num_arms)]
		self.theta_hat = [np.zeros(self.num_features).reshape((-1,1)) for i in range(self.num_arms)]


class ThompsonSampler:
	def __init__(self, num_arms, num_features, v_square):
		self.num_arms = num_arms
		self.num_features = num_features
		self.v_square = v_square
		self.B = [np.eye(num_features) for i in range(self.num_arms)]
		self.mu = [np.zeros((num_features, 1)) for i in range(self.num_arms)]
		self.f = [np.zeros((num_features, 1)) for i in range(self.num_arms)]

		self.B_inv = [np.eye(num_features) for i in range(self.num_arms)]

	def choose_arm(self, context):
		reward = []
		for i in range(self.num_arms):
			sample = np.random.multivariate_normal(self.mu[i].squeeze(), self.v_square * self.B_inv[i])
			reward.append(np.matmul(np.transpose(context), sample.reshape((-1,1)))[0][0])
		reward = np.array(reward)
		return reward.argmax()

	def update_param(self, context, arm, reward):
		self.B[arm] += np.matmul(context, np.transpose(context))
		self.B_inv[arm] = np.linalg.inv(self.B[arm])
		self.f[arm] += context * reward
		self.mu[arm] = np.matmul(self.B_inv[arm], self.f[arm])

	def reset(self):
		self.B = [np.eye(self.num_features) for i in range(self.num_arms)]
		self.mu = [np.zeros((self.num_features, 1)) for i in range(self.num_arms)]
		self.f = [np.zeros((self.num_features, 1)) for i in range(self.num_arms)]

		self.B_inv = [np.eye(self.num_features) for i in range(self.num_arms)]

class LinOracle:
	def __init__(self, num_arms, feature_list, true_cat):
		self.feature_list = feature_list
		self.num_arms = num_arms
		self.true_cat = np.repeat(np.array(true_cat).reshape((1,-1)), num_arms, axis=0) 
		self.model = [linear_model.LinearRegression(fit_intercept=False) for i in range(self.num_arms)]
		arm_array = np.array([0,1,2]).reshape((-1,1))
		self.cat = -((~(self.true_cat == arm_array)).astype(int))
		for i in range(self.num_arms):
			self.model[i].fit(feature_list, self.cat[i])

	def choose_arm(self, context):
		scores = []
		for i in range(self.num_arms):
			scores.append(self.model[i].predict(context.reshape((1,-1))).item())
		scores = np.array(scores)
		return scores.argmax()

	def calc_regret(self, context, arm):
		scores = []
		for i in range(self.num_arms):
			scores.append(self.model[i].predict(context.reshape((1,-1))).item())
		scores = np.array(scores)
		return np.amax(scores) - scores[arm]

	def update_param(self, context, arm, reward):
		pass
	def reset(self):
		pass

class SupervisedBandit:
	def __init__(self, feature_len):
		self.data = []
		self.model = linear_model.LinearRegression(fit_intercept=False)
		self.true_dosage = []
		self.feature_len = feature_len
		init_context = np.zeros(feature_len)
		self.model.fit(init_context.reshape((1,-1)), [35])

	def choose_arm(self, context):
		pred_dos = self.model.predict(context.reshape((1,-1))).item()
		if pred_dos < 21:
			return 0
		if pred_dos <= 49:
			return 1
		if pred_dos > 49:
			return 2

	def update_param(self, context, arm, true_dose):
		self.true_dosage.append(true_dose)
		self.data.append(context.reshape((1, -1)).squeeze(0))
		# print(self.data)
		self.model.fit(self.data, self.true_dosage)


	def reset(self):
		self.data = []
		self.model = linear_model.LinearRegression()
		self.true_dosage = []
		init_context = np.zeros(self.feature_len)
		self.model.fit(init_context.reshape((1,-1)), [35])




	