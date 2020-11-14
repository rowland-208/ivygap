#%%
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import TSNE
from tqdm import tqdm

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
for label, y in (
        ('All', y0),):
        #*((key, np.where(y0==key, key, 'Other')) for key in np.unique(y0))):
    #%%
    transcripts = []
    scores = []
    trials = 1000
    for seed in tqdm(range(trials)):
        clf = ExtraTreesClassifier(
            n_estimators=50, random_state=seed,
            max_depth=5, criterion='entropy',
            min_impurity_decrease=0.05)
        clf.fit(X, y)
        dimred = SelectFromModel(clf, prefit=True, max_features=50)

        transcripts.extend(ft.T.index[dimred.get_support()])
        scores.extend(clf.feature_importances_[dimred.get_support()])

    df0 = pd.DataFrame()
    df0['transcripts'] = transcripts
    df0['scores'] = scores

    #%%
    def f(gp):
        return pd.DataFrame(
            [[len(gp), max(gp['scores'])]],
            columns=['count', 'max'])
    df1 = df0.groupby('transcripts').apply(f).reset_index()
    df1['frequency'] = df1['count']/trials
    df1['selection'] = (df1['max']>0.019)|(df1['frequency']>0.1)
    fig1 = px.scatter(
        df1,
        x='frequency', y='max',
        hover_name='transcripts',
        color='selection',
        title='Monte Carlo results',
        labels={
            'frequency': 'Frequency',
            'max': 'Importance',
            'selection': 'Selected'})
    fig1.update_layout(
        xaxis=dict(range=[0,0.16]),
        yaxis=dict(range=[0.004,0.03]))

    df2 = df1[df1['selection']==True]
    df2.to_csv('site/montecarlo-expressions-{}_control.csv'.format(label))

    #%%
    mask = ft.T.index.isin(df2['transcripts'])
    X_new = X[:,mask]

    tsne = TSNE(
        n_components=3,
        n_iter=10000, learning_rate=20.,
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

    fig2 = px.scatter_3d(
        df, x='x1', y='x2', z='x3', 
        color='structure',
        hover_name='label')
    
    figures_to_html([fig1, fig2], 'site/montecarlo-{}_control.html'.format(label))