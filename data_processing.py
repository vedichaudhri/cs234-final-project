import pandas as pd
import numpy as np

def process_data(filename):
	data = pd.read_csv(filename)
	data = data[~data['Therapeutic Dose of Warfarin'].isnull()]
	features = process(data)
	age_arr = np.array([i for i in range(10)])

	age_col = features.Age.to_numpy()
	age_mat = (age_col[:,None]==age_arr).astype(int)
	feat_mat = features.to_numpy()
	bias_vec = np.ones(features.shape[0]).reshape((-1, 1))

	dosage = data['Therapeutic Dose of Warfarin'].to_numpy()
	feature_mat = np.concatenate((feat_mat, age_mat, bias_vec), axis=1)
	return feature_mat, features, dosage

def process(data):

	features = pd.DataFrame()
	data.loc[data.Race.isnull(), 'Race'] = "Unknown"
	features["White"] = (data.Race == "White").astype(int)
	features["Unknown"] = (data.Race == "Unknown").astype(int)
	features["Bl/AfrAmer"] = (data.Race == "Black or African American").astype(int)
	features["Asian"] = (data.Race == "Asian").astype(int)

	data.loc[data.Age == "0 - 9", 'Age'] = 0
	data.loc[data.Age == "10 - 19", 'Age'] = 1
	data.loc[data.Age == "20 - 29", 'Age'] = 2
	data.loc[data.Age == "30 - 39", 'Age'] = 3
	data.loc[data.Age == "40 - 49", 'Age'] = 4
	data.loc[data.Age == "50 - 59", 'Age'] = 5
	data.loc[data.Age == "60 - 69", 'Age'] = 6
	data.loc[data.Age == "70 - 79", 'Age'] = 7
	data.loc[data.Age == "80 - 89", 'Age'] = 8
	data.loc[data.Age == "90+", 'Age'] = 9
	data.loc[data.Age.isnull(), 'Age'] = 6

	features["Age"] = data.Age.astype(int)
	data = remove_na(data)

	features["Height"] = data["Height (cm)"]
	features["Weight"] = data["Weight (kg)"]
	features["Enzyme"] = data[['Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin']].any(axis=1).astype(int)
	features["Amiodarone"] = data['Amiodarone (Cordarone)'].astype(int)

	features["Diabetes"] = data["Diabetes"].astype(int)
	features["Heart_failure"] = data["Congestive Heart Failure and/or Cardiomyopathy"].astype(int)
	features["VKORC1_genotype_ag"] = (data["VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T"] == "A/G").astype(int)
	features["VKORC1_genotype_aa"] = (data["VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T"] == "A/A").astype(int)
	features["VKORC1_genotype_na"] = (data["VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T"].isnull()).astype(int)
	features["Cyp2C9_genotypes_12"] = (data["Cyp2C9 genotypes"] == '*1/*2').astype(int)
	features["Cyp2C9_genotypes_13"] = (data["Cyp2C9 genotypes"] == '*1/*3').astype(int)
	features["Cyp2C9_genotypes_22"] = (data["Cyp2C9 genotypes"] == '*2/*2').astype(int)
	features["Cyp2C9_genotypes_23"] = (data["Cyp2C9 genotypes"] == '*2/*3').astype(int)
	features["Cyp2C9_genotypes_33"] = (data["Cyp2C9 genotypes"] == '*3/*3').astype(int)
	features["Cyp2C9_genotypes_na"] = (data["Cyp2C9 genotypes"].isnull()).astype(int)
	features["Treatment_1"] = data["Indication for Warfarin Treatment"].map(lambda x: 1 if '1' in x else 0).astype(int)
	features["Treatment_2"] = data["Indication for Warfarin Treatment"].map(lambda x: 1 if '2' in x else 0).astype(int)
	features["Treatment_3"] = data["Indication for Warfarin Treatment"].map(lambda x: 1 if '3' in x else 0).astype(int)
	features["Treatment_4"] = data["Indication for Warfarin Treatment"].map(lambda x: 1 if '4' in x else 0).astype(int)
	features["Treatment_5"] = data["Indication for Warfarin Treatment"].map(lambda x: 1 if '5' in x else 0).astype(int)
	features["Treatment_6"] = data["Indication for Warfarin Treatment"].map(lambda x: 1 if '6' in x else 0).astype(int)
	features["Treatment_7"] = data["Indication for Warfarin Treatment"].map(lambda x: 1 if '7' in x else 0).astype(int)
	features["Treatment_8"] = data["Indication for Warfarin Treatment"].map(lambda x: 1 if '8' in x else 0).astype(int)

	return features

def remove_na(data):
	data["Diabetes"].fillna(value=0, inplace=True)
	data["Indication for Warfarin Treatment"].fillna(value='0', inplace=True)
	data["Congestive Heart Failure and/or Cardiomyopathy"].fillna(value=0, inplace=True)
	data['Carbamazepine (Tegretol)'].fillna(value = 0, inplace=True)
	data['Phenytoin (Dilantin)'].fillna(value = 0, inplace=True)
	data['Rifampin or Rifampicin'].fillna(value = 0, inplace=True)
	data['Amiodarone (Cordarone)'].fillna(value = 0, inplace=True)

	data['Height (cm)'].fillna(value = 168.047811, inplace=True)
	data['Weight (kg)'].fillna(value = 77.853057, inplace=True)

	return data
	

def get_truth_cat(arr):
	new_arr = np.zeros_like(arr)
	for i in range(len(arr)):
		if float(arr[i]) < 21:
			new_arr[i] = 0
		elif float(arr[i])>=21 and float(arr[i])<=49:
			new_arr[i] = 1
		else:
			new_arr[i] = 2
	new_arr = new_arr.astype(int)
	return new_arr
