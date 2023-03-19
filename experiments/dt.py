import graphviz
from numpy import mean
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from hdict import hdict
import matplotlib.pyplot as plt

# Load CSV
d = hdict.fromfile("all-except-VEP_N1.csv", fields=["df_"])
# print(d.df.keys())
df0: DataFrame = d.df_
attributes = [
    # "ID",
    "Delta", "Theta", "HighAlpha", "Beta", "Gamma", "Number segments",
    "2Hz pre-post wavelet change", "5Hz pre-post wavelet change", "12Hz pre-post wavelet change", "20Hz pre-post wavelet change", "30Hz pre-post wavelet change",
    "idade_crianca_meses_t1", "idade_crianca_meses_t2",
    "educationLevelAhmedNum_t1",
    "elegib2_t0", "elegib9_t0", "elegib14_t0", "risco_total_t0", "risco_class_t0", "ebia_tot_t1", "ebia_2c_t1", "ebia_tot_t2", "ebia_2c_t2",
    "epds_tot_t1", "epds_2c_t1", "epds_tot_t2", "epds_2c_t2", "pss_tot_t1", "mspss_tot_t1", "pss_2c_t1", "pss_tot_t2", "pss_2c_t2",
    "gad_tot_t1", "gad_2c_t1", "gad_tot_t2", "gad_2c_t2", "psi_pd_t1", "psi_pcdi_t1", "psi_dc_t1", "psi_tot_t1",
    # "final_8_t1",
    "bisq_3_mins_t1", "bisq_4_mins_t1", "bisq_9_mins_t1", "bisq_sleep_prob_t1", "bisq_sleep_prob_t2",
    # "final_10_t1", "final_10_t2",
    "chaos_tot_t1"
]
targets = [
    "bayley_1_t1", "bayley_2_t1", "bayley_3_t1", "bayley_6_t1", "bayley_16_t1", "bayley_7_t1", "bayley_17_t1", "bayley_18_t1", "bayley_8_t1", "bayley_11_t1", "bayley_19_t1", "bayley_12_t1",
    "bayley_20_t1", "bayley_21_t1", "bayley_13_t1", "bayley_22_t1", "bayley_23_t1", "bayley_24_t1", "bayley_1_t2", "bayley_2_t2", "bayley_3_t2", "bayley_6_t2", "bayley_16_t2", "bayley_7_t2", "bayley_17_t2", "bayley_18_t2", "bayley_8_t2", "bayley_11_t2", "bayley_19_t2",
    "bayley_12_t2", "bayley_20_t2", "bayley_21_t2", "bayley_13_t2", "bayley_22_t2", "bayley_23_t2", "bayley_24_t2",
    "ibq_sur_t1", "ibq_neg_t1", "ibq_reg_t1", "ibq_sur_t2", "ibq_neg_t2", "ibq_reg_t2"
]

df = df0[attributes]
for target in targets:
    # Clear NaNs by mean imputation
    df = df.fillna(df.mean())
    print(df.isna().sum())

    X = df
    y = df0[target] > mean(df0[target])
    y.replace(True, "high", inplace=True)
    y.replace(False, "low", inplace=True)

    model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=3,
        random_state=0,
        max_leaf_nodes=20
    )
    model = model.fit(X, y)
    dot_data = export_graphviz(model, out_file=None, feature_names=attributes, class_names=sorted(y.unique()), filled=True)
    graph = graphviz.Source(dot_data, filename=target, format="png")
    graph.render(view=False)
