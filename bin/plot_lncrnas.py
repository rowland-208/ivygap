#%%
import numpy as np
import pandas as pd
import plotly.express as px

# %%
nc = pd.read_csv(
    '~/personal/data/fastemc/2020-11-11/montecarlo-struct-features.txt',
    delimiter='\t')
nc = nc.sort_values('transcripts')
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

lncrna_transcrtips = nc[nc.gene_biotype=='lncRNA'].transcripts
mask = ft.columns.isin(lncrna_transcrtips)

X = X0[:,mask]

df = pd.DataFrame(
    (X - np.min(X, axis=0, keepdims=True))/
    (np.max(X, axis=0, keepdims=True) - np.min(X, axis=0, keepdims=True)),
    columns=nc[nc.gene_biotype=='lncRNA'].external_gene)

df['type'] = y0

# %%
df['ENSG00000248636'] = df['ENSG00000248636'] + 0.1
fig = px.scatter_3d(
    data_frame=df,
    x='ENSG00000250863',
    y='ENSG00000260244',
    z='ENSG00000261759',
    size='ENSG00000248636',
    color='type',
    )
fig.show()

#%%
fig = px.scatter_matrix(
    data_frame=df,
    dimensions=[col for col in df.columns if 'AC' in col],
    color='type')
fig.write_html('../site/lncrna-scatter.html')


# %%
