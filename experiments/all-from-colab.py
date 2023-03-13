"""
Temporary script. This might be split into many smaller scripts.
"""
import matplotlib as plt
import numpy as np
from google.colab import drive
import zipfile

from src.mlholdouts.misc import vstack_matrix, holdout_framework

# drive.mount('/content/drive')


archive = zipfile.ZipFile('./drive/MyDrive/Matrices/Fujita_EEG_request_15.09.22.zip', 'r')
# extract all matrices
archive.extractall()

import pandas as pd

df = pd.read_csv(r'./drive/MyDrive/Dados Comportamentais e Demográficos/grover160123.csv')
df.head()





from os.path import exists

matrix_folder_path = "./Fujita_EEG_request_15.09.22/Connectivity Matrices/"
exists_matrix_data = []
matrix_data_alpha = None
matrix_data_beta = None
matrix_data_theta = None

all_id_estudo = df["id_estudo"]
for idx_subject in range(df.shape[0]):
    id_estudo = all_id_estudo[idx_subject]
    if id_estudo < 10:
        id_estudo = "00" + str(id_estudo)
    elif id_estudo < 100:
        id_estudo = "0" + str(id_estudo)
    else:
        id_estudo = str(id_estudo)

    alpha_file_name = "Alpha/p" + id_estudo + "_3m_RS_dwPLI_alpha.txt"
    beta_file_name = "Beta/p" + id_estudo + "_3m_RS_dwPLI_Beta.txt"
    theta_file_name = "Theta/p" + id_estudo + "_3m_RS_dwPLI_theta.txt"
    # verify if connectivity matrices exists
    path_to_file_alpha = matrix_folder_path + alpha_file_name
    path_to_file_beta = matrix_folder_path + beta_file_name
    path_to_file_theta = matrix_folder_path + theta_file_name

    file_exists_alpha = exists(path_to_file_alpha)
    file_exists_beta = exists(path_to_file_beta)
    file_exists_theta = exists(path_to_file_theta)

    # file only exists if it has all three matrices
    file_exists = file_exists_alpha and file_exists_beta and file_exists_theta

    exists_matrix_data.append(file_exists)
    if file_exists:
        matrix_alpha = np.loadtxt(path_to_file_alpha)
        upp_matrix_alpha = matrix_alpha[np.triu_indices(matrix_alpha.shape[0], k=1)]
        matrix_beta = np.loadtxt(path_to_file_beta)
        upp_matrix_beta = matrix_beta[np.triu_indices(matrix_beta.shape[0], k=1)]
        matrix_theta = np.loadtxt(path_to_file_theta)
        upp_matrix_theta = matrix_alpha[np.triu_indices(matrix_theta.shape[0], k=1)]

        matrix_data_alpha = vstack_matrix(matrix_data_alpha, upp_matrix_alpha)
        matrix_data_beta = vstack_matrix(matrix_data_beta, upp_matrix_beta)
        matrix_data_theta = vstack_matrix(matrix_data_theta, upp_matrix_theta)

# Bayley Composto
# bayley_3_t1 -> composite cognitive
# bayley_8_t1 -> composite language
# bayley_13_t1 -> composite motor
# bayley_24_t1 -> composite socioemotional
# Bayley Bruto
# bayley_1_t1 -> raw cognitive
# bayley_18_t1 -> raw language
# bayley_21_t1 -> raw motor
# bayley_22_t1 -> socioemotional
# elegib14_t0 -> sexo

# brain data
# bayley_rows = ["bayley_3_t1","bayley_8_t1","bayley_13_t1","bayley_24_t1"]
# bayley_rows = ["bayley_1_t1","bayley_18_t1","bayley_21_t1","bayley_22_t1"]#,"elegib14_t0"] # good results
# ibq_sur_t1 -> Surgency/Extraversion
# ibq_neg_t1 -> Negative Affectivity
# ibq_reg_t1 -> Orienting/Regulation
bayley_rows = ["ibq_sur_t1", "ibq_neg_t1", "ibq_reg_t1"]

bayley_data = df[exists_matrix_data]
bayley_data = bayley_data[bayley_rows]
bayley_valid_rows = bayley_data.isna().any(axis=1)
bayley_data.shape
bayley_data.head()

# biological/enviromental data
# elegib14_t0 -> sexo
environment_rows = ["elegib14_t0", "risco_total_t0", "ebia_tot_t1", "pss_tot_t1", "epds_tot_t1", "gad_tot_t1", "psi_tot_t1", "bisq_sleep_prob_t1"]
env_data = df[exists_matrix_data]
env_data = env_data[environment_rows]
env_valid_rows = env_data.isna().any(axis=1)
env_data.shape

# select kids with enviromental and bayley data
valid_rows = np.logical_or(bayley_valid_rows, env_valid_rows)
#
bayley_data_valid = bayley_data[np.logical_not(valid_rows)]
env_data_valid = env_data[np.logical_not(valid_rows)]
env_data_valid.loc[:, "elegib14_t0"] = env_data_valid.loc[:, "elegib14_t0"] - 1  # sex to have 0/1 values only

# select eeg spectrum of valid individuals
X_alpha = matrix_data_alpha[np.logical_not(valid_rows)]
X_beta = matrix_data_beta[np.logical_not(valid_rows)]
X_theta = matrix_data_theta[np.logical_not(valid_rows)]

print(bayley_data_valid.shape)
print(X_alpha.shape)
print(X_beta.shape)
print(X_theta.shape)
print(env_data_valid.shape)
bayley_data_valid.head()

env_data_valid.shape
env_data_valid.head()

np.savetxt('X_alpha.csv', X_alpha, delimiter=',', fmt='%f')
np.savetxt('X_beta.csv', X_beta, delimiter=',', fmt='%f')
np.savetxt('X_theta.csv', X_theta, delimiter=',', fmt='%f')
np.savetxt('X_env.csv', np.array(env_data_valid), delimiter=',', fmt='%f')
np.savetxt('Y.csv', np.array(bayley_data_valid), delimiter=',', fmt='%f')

X_env = np.array(env_data_valid)

X_ddpls = {0: X_alpha, 1: X_beta, 2: X_theta, 3: X_env}  # {0:X_alpha,1:X_env} #
X = np.hstack([X_alpha, X_beta, X_theta, X_env])
X_alpha_env = np.hstack([X_alpha, X_env])

Y = np.array(bayley_data_valid)

print(X_env.shape)
print(X_alpha_env.shape)
print(X.shape)

from scipy.stats.stats import PearsonRConstantInputWarning
import random
#
import warnings

warnings.filterwarnings("ignore", category=PearsonRConstantInputWarning)

random.seed(42)
res_ddspls_brain = holdout_framework(X_alpha, Y, method="ddspls", stability=True)
print(res_ddspls_brain)

random.seed(42)
res_ddspls_env = holdout_framework(X_env, Y, method="ddspls", stability=True)
print(res_ddspls_env)

random.seed(42)
res_ddspls = holdout_framework(X_ddpls, Y, method="ddspls", stability=True)
print(res_ddspls)

env_bio_coeff = res_ddspls["model"].B[1]
# print(brain_coeff)

for idx_measure in range(len(bayley_rows)):
    print(bayley_rows[idx_measure] + ":")
    for idx_env_var in range(env_bio_coeff.shape[0]):
        print("\t" + environment_rows[idx_env_var] + " : " + str(env_bio_coeff[idx_env_var, idx_measure]))

print(res_ddspls["model"].R)

# save lasso weights as a matrix
n = 98
for idx_measure in range(len(bayley_rows)):
    B = np.zeros((n, n))
    idx = 0
    brain_coeff = res_ddspls["model"].B[0][:, idx_measure]
    bool_coeff = (brain_coeff != 0)
    print(np.sum(bool_coeff))
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (np.abs(brain_coeff[idx]) > 1e-3):
                B[i, j] = B[j, i] = brain_coeff[idx]
            idx = idx + 1
    np.savetxt('ddspls_weights_' + bayley_rows[idx_measure] + '_env.edge', B, delimiter='\t')

random.seed(42)
res_lasso = holdout_framework(X_alpha_env, Y[:, 2], method="lasso", stability=True)
print(res_lasso)

best_coeff = res_lasso["model"].coef_
brain_coeff = best_coeff[0:X_alpha.shape[1]]
env_bio_coeff = best_coeff[X_alpha.shape[1]:X_alpha_env.shape[1]]
for idx_env_var in range(len(environment_rows)):
    print(environment_rows[idx_env_var] + " : " + str(env_bio_coeff[idx_env_var]))

# save lasso weights as a matrix
n = 98
B = np.zeros((n, n))
idx = 0
for i in range(n - 1):
    for j in range(i + 1, n):
        B[i, j] = B[j, i] = brain_coeff[idx]
        idx = idx + 1
np.savetxt('lasso_weights_alpha_env.edge', B, delimiter='\t')

import matplotlib.image as mpimg

first_png = mpimg.imread('first.png')
second_png = mpimg.imread('second.png')
third_png = mpimg.imread('third.png')
fourth_png = mpimg.imread('fourth.png')
#
# plt.figure(figsize=(20, 20))
ig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].imshow(first_png)
# axs[0, 0].set_title('Axis [0, 0]')
axs[0, 1].imshow(second_png)
axs[1, 0].imshow(third_png)
axs[1, 1].imshow(fourth_png)
