import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
df0 = pd.read_csv('data0.csv', names=['square', 'price'])
sns.lmplot('square', 'price', df0, height=6, fit_reg=True)