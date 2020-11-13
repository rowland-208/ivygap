import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import TSNE

ft = pd.read_csv(
    '~/personal/data/fastemc/2020-11-11/_structNormalizedExp.txt',
    delimiter='\t').T
lb = pd.read_excel(
    '~/personal/data/fastemc/2020-11-11/columnData_complete.xlsx'
    )
lb.quant_sf = lb.quant_sf.apply(lambda s: '_'.join(s.split('_')[:4]))
lb = lb.set_index('quant_sf')

X = ft.values
y0 = ft.index.map(lb.structure_name).values
for label, y in (
        ('All', y0),
        *((key, np.where(y0==key, key, 'Other')) for key in np.unique(y0))):

    clf = ExtraTreesClassifier(
        n_estimators=500, random_state=2,
        max_depth=10, criterion='entropy',
        min_impurity_decrease=0.05)
    clf.fit(X, y)
    print('Score: ', clf.score(X, y))

    dimred = SelectFromModel(clf, prefit=True, max_features=50)
    X_new = dimred.transform(X)

    tsne = TSNE(
        n_components=3,
        n_iter=10000, learning_rate=40.,
        early_exaggeration=20.,
        perplexity=30., random_state=0)
    X_3d = tsne.fit_transform(X_new, y)
    X_min = np.min(X_3d, axis=0)
    X_max= np.max(X_3d, axis=0)
    X_3d = (X_3d - X_min)/(X_max - X_min)

    df = pd.DataFrame(
        np.transpose([
            ft.index,
            ft.index.map(lb.structure_name),
            *(X_3d.T)]),
        columns=['label', 'structure', 'x1', 'x2', 'x3'])

    fig = px.scatter_3d(
        df, x='x1', y='x2', z='x3', 
        color='structure',
        hover_name='label')
    fig.write_html('structures-{}.html'.format(label))

    with open('expressions-{}.csv'.format(label), 'w') as fp:
        fp.write(','.join(ft.T.index[dimred.get_support()]))
