import pathlib
import statistics as stat
import numpy as np
import pandas as pd
import plotly.express as px
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from app import app

#
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
#
# Import data
#
df_fresh = pd.read_pickle(DATA_PATH.joinpath('peel_fersk'))
df_6mnd = pd.read_pickle(DATA_PATH.joinpath('peel_6mnd'))
df = pd.read_pickle(DATA_PATH.joinpath('peel_all_coded'))
df_decompose = pd.DataFrame()
print(df_decompose)
#
#
#
#####
#####
expl_var = []
tape_set = list(np.unique(list(df['name'])))
substrate_set = list(np.unique(list(df['substrate'])))
features = ['T-peel',
            'K_ini',
            'NRMSE_mean',
            'E',
            'Density',
            'Surf Energy',
            'age'
            ]
####
RMSE_mean = []
NRMSE_mean = []
Peel_cov = []
for i in range(len(df)):
    df['T-peel'] = df['T-peel'].replace(df['T-peel'].iloc[i], round(df['T-peel'].iloc[i], 2))
    df['T-peel stdv'] = df['T-peel stdv'].replace(df['T-peel stdv'].iloc[i], round(df['T-peel stdv'].iloc[i], 2))
    df['Peel avg'] = df['Peel avg'].replace(df['Peel avg'].iloc[i], round(df['Peel avg'].iloc[i], 2))
    df['K_ini'] = df['K_ini'].replace(df['K_ini'].iloc[i], round(df['K_ini'].iloc[i], 2))
    df['K_ini_stdv'] = df['K_ini_stdv'].replace(df['K_ini_stdv'].iloc[i], round(df['K_ini_stdv'].iloc[i], 2))
    df['Peak F'] = df['Peak F'].replace(df['Peak F'].iloc[i], round(df['Peak F'].iloc[i], 2))
    if df['T-peel'].iloc[i] != 0:
        Peel_cov.append(round(df['T-peel stdv'].iloc[i]/df['T-peel'].iloc[i], 2))
    else:
        Peel_cov.append(0)
    if df['RMSE'].iloc[i] != 0:
        RMSE_mean.append(round(stat.mean(df['RMSE'].iloc[i]), 2))
        NRMSE_mean.append(round(stat.mean(df['NRMSE'].iloc[i]), 2))
    else:
        RMSE_mean.append(0)
        NRMSE_mean.append(0)
df['RMSE_mean'] = RMSE_mean
df['NRMSE_mean'] = NRMSE_mean
df['Peel COV'] = Peel_cov
df = df[df['test type'] == 't-peel'][['No', 'name', 'substrate', 'test type', 'load procedure', 'age',
       'conditioning time', 'T-peel', 'T-peel stdv', 'Peel COV', 'Peel avg', 'Peel avg stdev', 'K_ini', 'K_ini_stdv',
       'Peel LR', 'Peak F', 'Peak F stdev', 'min loc', 'min loc stdev', 'RMSE_mean', 'NRMSE_mean', 'envir',
       'max loc', 'max loc stdev', 'filename', 'E', 'Density', 'Surf Energy']]
print('length of df: ', len(df))
#
#
#
# App layout
#
#
#
layout = html.Div([
    html.Br(),
    html.Label([' Tighten / SINTEF '],
               style={'font-size': 10, 'font-family': 'Verdana', "text-align": "center"}),
    html.Br(),
    html.Br(),
    html.Label(['Principal Component Analysis is a tool in explanatory data analysis,'
                ' that allows for eigenvalue decomposition of data and changing the data base accordingly.'],
               style={'font-size': 16, 'font-family': 'Verdana', "text-align": "center"}),
    html.Br(),
    html.Label([' The computed principal components can be selected in smaller number than original dimension of data '
                ' to reduce data dimensionality, allow 2D/3D visualization or simplify data for further analysis.'],
               style={'font-size': 16, 'font-family': 'Verdana', "text-align": "center"}),
    html.Br(),
    html.Label([' The number of parameters explaining data are called features and each has a physical meaning. '
                ' Principal components are linear combinations of features that carry no such physical meaning.'],
               style={'font-size': 16, 'font-family': 'Verdana', "text-align": "center"}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Label([' Select at least three features of data to include in principle component analysis: '
                ],
               style={'font-size': 12, 'font-family': 'Verdana', "text-align": "center"}),
    html.Br(),
    dcc.Dropdown(id='features',
                 options=[{'label': x, 'value': x} for x in
                          features],
                 value=['T-peel', 'age', 'Surf Energy'],
                 multi=True,
                 disabled=False,
                 clearable=True,
                 searchable=True,
                 placeholder='Choose feature...',
                 className='form-dropdown',
                 style={'width': "60%"},
                 persistence='string',
                 persistence_type='memory'),
    html.Br(),
    html.Label([' Select at least one tape type to include in principle component analysis: '
                ],
               style={'font-size': 12, 'font-family': 'Verdana', "text-align": "center"}),
    dcc.Dropdown(id='tapes',
                 options=[{'label': x, 'value': x} for x in
                          tape_set],
                 value=['A', 'B', 'C', 'D'],
                 multi=True,
                 disabled=False,
                 clearable=True,
                 searchable=True,
                 placeholder='Choose tape...',
                 className='form-dropdown',
                 style={'width': "60%"},
                 persistence='string',
                 persistence_type='memory'),
    html.Br(),
    html.Label([' Select at least one substrate to include in principle component analysis: '
                ],
               style={'font-size': 12, 'font-family': 'Verdana', "text-align": "center"}),
    dcc.Dropdown(id='substrates',
                 options=[{'label': x, 'value': x} for x in
                          substrate_set],
                 value=['tre', 'stål', 'hdpesmooth'],
                 multi=True,
                 disabled=False,
                 clearable=True,
                 searchable=True,
                 placeholder='Choose substrate...',
                 className='form-dropdown',
                 style={'width': "60%"},
                 persistence='string',
                 persistence_type='memory'),
    dash_table.DataTable(data=df_decompose.to_dict('records')),
    html.Br(),
    html.Br(),
    html.Label([' Variance is explained by principal components PC-1, PC-2 and PC-3: '
                ],
               style={'font-size': 14, 'font-family': 'Verdana', "text-align": "center"}),
    html.Div(id='variance'),
    html.Br(),
    html.Table([
        html.Tr([html.Td('feature'), html.Td(id='feature')]),
        html.Tr([html.Td('PC-1'), html.Td(id='decomp_PC1')]),
        html.Tr([html.Td('PC-2'), html.Td(id='decomp_PC2')]),
        html.Tr([html.Td('PC-3'), html.Td(id='decomp_PC3')]),
    ]),
    html.Br(),
    dcc.Graph(id='PCA-graph', style={'width': '200vh', 'height': '90vh'}),
    html.Br(),
    html.Br(),
])


# -------------------------------------------------------------------------------------
# Create scatter chart
@app.callback(
    Output('PCA-graph', 'figure'),
    Output('variance', 'children'),
    Output('decomp_PC1', 'children'),
    Output('decomp_PC2', 'children'),
    Output('decomp_PC3', 'children'),
    Output('feature', 'children'),
    [Input('features', 'value'),
     Input('substrates', 'value'),
     Input('tapes', 'value')
     ]
)
def build_graph2(features, substrate_set, tape_set):
    #
    #
    print('length of df: ', len(df))
    print(tape_set, type(tape_set))
    print(substrate_set, type(substrate_set))
    df_PCA = df[df['name'].isin(tape_set)][['T-peel', 'name', 'substrate',
                                            'K_ini', 'NRMSE_mean', 'age',
                                            'T-peel stdv', 'K_ini_stdv', 'envir', 'No',
                                            'E', 'Density', 'Surf Energy'
                                            ]]
    print('length of df_PCA: ', len(df_PCA))
    df_PCA = df_PCA[df_PCA['substrate'].isin(substrate_set)]
    print('length of df_PCA: ', len(df_PCA))
    print(df_PCA)
    df_PCA = df_PCA.reset_index()
    ###################
    # data standarizing
    # Separating out the features
    #
    x = df_PCA.loc[:, features].values
    # Separating out the target
    y = df_PCA.loc[:, ['name']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    # PCA to 3D
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['PC 1',
                                        'PC 2',
                                        'PC 3',
                                        # 'principal component 4',
                                        ])
    df_decompose = pd.DataFrame(pca.components_,
                                index=['PC-1',
                                       'PC-2',
                                       'PC-3',
                                       # 'PC-4',
                                       ],
                                columns=features)
    decomp_1 = [round(k, 2) for k in list(df_decompose.iloc[0])]
    decomp_2 = [round(k, 2) for k in list(df_decompose.iloc[1])]
    decomp_3 = [round(k, 2) for k in list(df_decompose.iloc[2])]
    finalDf = pd.concat([principalDf, df_PCA[['name', 'substrate', 'envir']]], axis=1)
    #print(principalDf, finalDf)
    #print(df_PCA)
    ### explained variance
    print('explained variance as ratio: ', pca.explained_variance_ratio_,
          'sum: ', sum(pca.explained_variance_ratio_))
    expl_var = [round(k, 2) for k in pca.explained_variance_ratio_]
    #
    # figure PCA
    #
    ##########
    fig = px.scatter_3d(finalDf,
                        x='PC 1',
                        y='PC 2',
                        z='PC 3',
                        hover_name='envir',
                        color='name',
                        color_discrete_sequence=['red', 'green', 'blue', 'black', 'magenta', 'gray',
                                                 'dimgray', 'lightgray', 'pink', 'orange',
                                                 'lavender', 'seagreen', 'navy', 'olive', 'azure',
                                                 'gold', 'hotpink', 'magenta', 'lawngreen',
                                                 'tan', 'plum', 'chocolate', 'teal', 'lime'
                                                 ],
                        size_max=6,
                        symbol='substrate',
                        symbol_sequence=[
                            'circle', 'circle-open',
                            'square', 'square-open',
                            'diamond', 'diamond-open',
                            'cross', 'x'],
                        opacity=0.8
                        )
    return fig, f'explained variance ratio: {expl_var}, sum: {round(sum(expl_var)*100,2)} % of variance is explained', str(decomp_1), str(decomp_2), str(decomp_3), str(features)

