import pandas as pd
import numpy as np

class FixedBaseline:

	def __init__(self):
		pass

	def choose_arm(self, context):
		return 1
	def update_param(self, context, arm, reward):
		pass
	def reset(self):
		pass

class ClinicalBaseline:

	def __init__(self):
		pass

	def choose_arm(self, context):
		w = context[0]
		un = context[1]
		bl = context[2]
		asi = context[3]
		age = context[4]
		height = context[5]
		weight = context[6]
		enzyme = context[7]
		amiodarone = context[8]
		score = 4.0376 - 0.2546*age + 0.0118*height + 0.0134* weight + 1.2799* enzyme - 0.5695*amiodarone
		score += (-0.6752*asi +0.4060*bl  + 0.0443*un)
		score = (score**2)[0]
		if score<21:
			return 0
		if score <= 49:
			return 1
		return 2

	def update_param(self, context, arm, reward):
		pass
	def reset(self):
		pass

