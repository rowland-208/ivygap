#%%
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import TSNE

#%%
# https://stackoverflow.com/questions/46821554/multiple-plotly-plots-on-1-page-without-subplot?rq=1
def figures_to_html(figs, filename):
    '''Saves a list of plotly figures in an html file.

    Parameters
    ----------
    figs : list[plotly.graph_objects.Figure]
        List of plotly figures to be saved.

    filename : str
        File name to save in.

    '''
    import plotly.offline as pyo

    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")

    add_js = True
    for fig in figs:

        inner_html = pyo.plot(
            fig, include_plotlyjs=add_js, output_type='div'
        )

        dashboard.write(inner_html)
        add_js = False

    dashboard.write("</body></html>" + "\n")

#%%
ft = pd.read_csv(
    '~/personal/data/fastemc/2020-11-11/_structNormalizedExp.txt',
    delimiter='\t').T
lb = pd.read_excel(
    '~/personal/data/fastemc/2020-11-11/columnData_complete.xlsx'
    )
lb.quant_sf = lb.quant_sf.apply(lambda s: '_'.join(s.split('_')[:4]))
lb = lb.set_index('quant_sf')

X = ft.values
X = np.random.permutation(X)
X = np.random.permutation(X.T).T
y0 = ft.index.map(lb.structure_name).values

#%%
x = []
y = []
for d in np.arange(1, 20):
    clf = ExtraTreesClassifier(
        n_estimators=50, random_state=2,
        max_depth=d, criterion='entropy',
        min_impurity_decrease=0.05)
    clf.fit(X, y0)
    score = clf.score(X, y0)
    x.append(d)
    y.append(score)
fig1 = px.scatter(
    x=x, y=y,
    labels={
        'x': 'Max Depth',
        'y': 'Accuracy'},
    title='Num Estimators=50, Min Impurity=0.05')
fig1.update_layout(
    xaxis=dict(range=[0, 21]),
    yaxis=dict(range=[-0.05,1.05]))

#%%
x = []
y = []
for n in np.arange(10, 301, 10):
    clf = ExtraTreesClassifier(
        n_estimators=n, random_state=2,
        max_depth=5, criterion='entropy',
        min_impurity_decrease=0.05)
    clf.fit(X, y0)
    score = clf.score(X, y0)
    x.append(n)
    y.append(score)
fig2 = px.scatter(
    x=x, y=y,
    labels={
        'x': 'Num Estimators',
        'y': 'Accuracy'},
    title='Min Impurity=0.05, Max Depth=5')
fig2.update_layout(
    xaxis=dict(range=[0, 310]),
    yaxis=dict(range=[-0.05,1.05]))

#%%
x = []
y = []
for imp in np.linspace(0, 0.2, 30):
    clf = ExtraTreesClassifier(
        n_estimators=50, random_state=2,
        max_depth=5, criterion='entropy',
        min_impurity_decrease=imp)
    clf.fit(X, y0)
    score = clf.score(X, y0)
    x.append(imp)
    y.append(score)
fig3 = px.scatter(
    x=x, y=y,
    labels={
        'x': 'Min Impurity',
        'y': 'Accuracy'},
    title='Num Estimator=50, Max Depth=5')
fig3.update_layout(
    xaxis=dict(range=[-0.01,0.21]),
    yaxis=dict(range=[-0.05,1.05]))

#%%
figures_to_html([fig1, fig2, fig3], 'site/meta-analysis_control.html')