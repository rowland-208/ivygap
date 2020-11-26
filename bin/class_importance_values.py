#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import TSNE
from tqdm import tqdm

#%%
ft = pd.read_csv(
    '~/personal/data/fastemc/2020-11-11/_structNormalizedExp.txt',
    delimiter='\t').T
lb = pd.read_excel(
    '~/personal/data/fastemc/2020-11-11/columnData_complete.xlsx'
    )
lb.quant_sf = lb.quant_sf.apply(lambda s: '_'.join(s.split('_')[:4]))
lb = lb.set_index('quant_sf')

X0 = ft.values
y0 = ft.index.map(lb.structure_name).values

sl = pd.read_csv('../data/montecarlo-expressions-All.csv')

mask = ft.columns.isin(sl.transcripts)

X = X0[:,mask]

# %%
seed = 0
for label, y in (
        (key, np.where(y0==key, key, 'Other')) for key in np.unique(y0)):
    seed = 0
    clf = ExtraTreesClassifier(max_depth=2, n_estimators=1000, random_state=seed)
    clf.fit(X, y)
    print(clf.score(X, y))
    sl['{}_importance'.format(label)] = clf.feature_importances_

#%%
sl.to_csv('../data/montecarlo-expression-All.csv')

#%%
df = pd.DataFrame()
df['Feature Importance'] = pd.concat([
    _sl[key]
    for key in _sl.columns
    if 'importance' in key])
df['Transcript'] = pd.concat([
    _sl.transcripts
    for key in _sl.columns
    if 'importance' in key])
df['Type'] = pd.concat([
    pd.Series([key.replace('_importance', '') for _ in _sl.index])
    for key in _sl.columns
    if 'importance' in key])

fig = px.scatter(
    data_frame=df,
    x='Transcript',
    y='Feature Importance',
    color='Type')
fig.show()
fig.write_html('../site/feature-importance.html')

