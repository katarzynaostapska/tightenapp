import pathlib
import statistics as stat
import pandas as pd
import plotly as plt
import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from app import app

#
# get relative data folder
#
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
#
# Import data
#
df_fresh = pd.read_pickle(DATA_PATH.joinpath('peel_fersk'))
df_6mnd = pd.read_pickle(DATA_PATH.joinpath('peel_6mnd'))
df = pd.read_pickle(DATA_PATH.joinpath('peel_all_coded'))
#
#
#
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
df = df[df['test type'] == 't-peel']
#[['No', 'tape', 'substrate', 'test type', 'load procedure', 'age',
#                'conditioning time', 'T-peel', 'T-peel stdv', 'Peel COV',
#                'Peel avg', 'Peel avg stdev', 'K_ini', 'K_ini_stdv',
#                'Peel LR', 'Peak F', 'Peak F stdev', 'min loc',
#                'min loc stdev', 'RMSE_mean', 'NRMSE_mean', 'envir',
#                'max loc', 'max loc stdev', 'filename']]

# Creating an ID column name gives us more interactive capabilities
df.index.name = 'id'

# -------------------------------------------------------------------------------------
# App layout
#
layout = html.Div([
    html.Br(),
    html.Label([' Tighten / SINTEF '],
               style={'font-size': 10, 'font-family': 'Verdana', "text-align": "center"}),
    html.Br(),
    html.Label(['The graph below depicts variation of peel resistance '
                'for a given tape on a given substrate before and after aging '
                'with breakdown on number of specimens. '
                'Whenever fresh tape series contain 5 specimens, the data for sub-samples of 3, 4 and 2 were computed '
                'to allow comparison with aged tape series containing only 2 samples in most cases. '
                'The down-sampling procedure considers all subsets of 4, 3, or 2 out of 5 as shown in figure below. '
                'Among the resulting subsets, the variation and mean envelopes together with median are computed.'],
               style={"text-align": "center"}),
    html.Img(src=app.get_asset_url('sampling.png'), height=300),
    html.Br(),
    html.Br(),
    html.Div(children=[
        html.Label(['Choose tape 1:'], style={'font-weight': 'bold', 'display': 'inline-block', "margin-left": "25px"}),
        html.Label(['Choose tape 2:'], style={'font-weight': 'bold', 'display': 'inline-block', "margin-left": "425px"}),
        html.Label(['Choose tape 3:'], style={'font-weight': 'bold', 'display': 'inline-block', "margin-left": "625px"}),
            ]),
    html.Div(children=[
        dcc.Dropdown(id='tape_1',
                 options=[{'label': x, 'value': x} for x in
                          df.sort_values('name')['name'].unique()],
                 value='A',
                 multi=False,
                 disabled=False,
                 clearable=True,
                 searchable=True,
                 placeholder='Choose tape...',
                 className='form-dropdown',
                 style={'width': "30%", 'display': 'inline-block'},
                 persistence='string',
                 persistence_type='memory'),
        dcc.Dropdown(id='tape_2',
                 options=[{'label': x, 'value': x} for x in
                          df.sort_values('name')['name'].unique()],
                 value='B',
                 multi=False,
                 disabled=False,
                 clearable=True,
                 searchable=True,
                 placeholder='Choose tape...',
                 className='form-dropdown',
                 style={'width': "30%", 'display': 'inline-block'},
                 persistence='string',
                 persistence_type='memory'),
        dcc.Dropdown(id='tape_3',
                 options=[{'label': x, 'value': x} for x in
                          df.sort_values('name')['name'].unique()],
                 value='C',
                 multi=False,
                 disabled=False,
                 clearable=True,
                 searchable=True,
                 placeholder='Choose tape...',
                 className='form-dropdown',
                 style={'width': "30%", 'display': 'inline-block'},
                 persistence='string',
                 persistence_type='memory'),
        ]),
    html.Div(children=[
        dcc.Graph(id='our_graph', style={'display': 'inline-block'}),
        dcc.Graph(id='our_graph1', style={'display': 'inline-block'}),
        dcc.Graph(id='our_graph2', style={'display': 'inline-block'})
    ])
])


# -------------------------------------------------------------------------------------
# Create scatter chart
@app.callback(
    Output('our_graph', 'figure'),
    [Input('tape_1', 'value')
     ]
)
def build_graph(tape_1):
    dff = df[df['name'] == tape_1]
    substrate0 = dff[dff['age'] == 0][['substrate']]
    substrate6_ute = dff[(dff['age'] == 6) & (dff['envir'] == 'ute')][['substrate']]
    substrate6_inne = dff[(dff['age'] == 6) & (dff['envir'] == 'inne')][['substrate']]
    dff = dff.sort_values(by=['envir'])
    ###
    x_2 = dff[['TP_stmean2_max', 'TP_stmean2_min', 'TP_stmean2_medi', 'TP_stdev2_min', 'TP_stdev2_max']]
    x_3 = dff[['TP_stmean3_max', 'TP_stmean3_min', 'TP_stmean3_medi', 'TP_stdev3_min', 'TP_stdev3_max']]
    x_4 = dff[['TP_stmean4_max', 'TP_stmean4_min', 'TP_stmean4_medi', 'TP_stdev4_min', 'TP_stdev4_max']]
    x_5_limax = dff['T-peel'] + dff['T-peel stdv']
    x_5_limin = dff['T-peel'] - dff['T-peel stdv']
    x_5 = dff[['T-peel']]
    x_5['T-peel_max'] = x_5_limax
    x_5['T-peel_min'] = x_5_limin
    ###

    fig = plt.subplots.make_subplots(rows=len(substrate0), cols=1,
                        subplot_titles=(list(substrate0['substrate'])),
                        vertical_spacing=0.02,
                        shared_xaxes=True)
    for k in range(len(substrate0)):
        if dff['No'].iloc[k] == 2:
            fig.append_trace(go.Scatter(x=x_2.iloc[k], y=[2] * 5, line=dict(color='red', width=1), name='N = 2',
                                        text=['COV:' + str(round((x_2.iloc[k][4] - x_2.iloc[k][2]) / x_2.iloc[k][2], 2))],
                                        textposition="bottom center", mode="markers+lines+text"),
                             row=k + 1, col=1)
        else:
            fig.append_trace(
                go.Scatter(x=x_2.iloc[k], y=[2] * 5, line=dict(color='red', width=1), name='N = 2 (down-sampled)',
                           text=['COV:'+str(round((x_2.iloc[k][4]-x_2.iloc[k][2])/x_2.iloc[k][2], 2))],
                           textposition="bottom center", mode="markers+lines+text"),
                row=k + 1, col=1)
            fig.append_trace(go.Scatter(x=x_3.iloc[k], y=[3] * 5, line=dict(color='royalblue', width=1),
                                        name='N = 3 (down-sampled)',
                                        text=['COV:' + str(round((x_3.iloc[k][4]-x_3.iloc[k][2])/x_3.iloc[k][2], 2))],
                                        textposition="bottom center", mode="markers+lines+text"
                                        ),
                             row=k + 1, col=1)
            fig.append_trace(
                go.Scatter(x=x_4.iloc[k], y=[4] * 5, line=dict(color='green', width=1), name='N = 4 (down-sampled)',
                           text=['COV:' + str(round((x_4.iloc[k][4] - x_4.iloc[k][2]) / x_4.iloc[k][2], 2))],
                           textposition="bottom center", mode="markers+lines+text"
                           ),
                row=k + 1, col=1)
            fig.append_trace(go.Scatter(x=x_5.iloc[k], y=[4.9] * 3, line=dict(color='yellow', width=1), name='N = 5',
                                        text=['COV:'+str(round((x_5.iloc[k][1]-x_5.iloc[k][0])/x_5.iloc[k][0], 2))],
                                        textposition="bottom center", mode="markers+lines+text"
                                        ),
                             row=k + 1, col=1)
    for j in range(len(substrate6_ute)):
        fig.append_trace(go.Scatter(x=x_5.iloc[len(substrate0) + j], y=[dff['No'].iloc[k]+0.1] * 5,
                                    line=dict(color='black', width=1),
                                    name='aged ute: N = '+str(dff['No'].iloc[k]),
                                    text=['ut:' + str(round((x_5.iloc[len(substrate0)
                                                              + j][1]-x_5.iloc[len(substrate0)+j][0])/x_5.iloc[len(substrate0)+j][0], 2))],
                                    textposition="middle center", mode="markers+lines+text"
                                    ),
                         row=j + 1, col=1)
    for k in range(len(substrate6_inne)):
        fig.append_trace(go.Scatter(x=x_5.iloc[len(substrate0) + len(substrate6_ute)+k], y=[dff['No'].iloc[k]+0.2]*5,
                                    line=dict(color='gray', width=1),
                                    name='aged inne: N = '+str(dff['No'].iloc[k]),
                                    text=['in:' + str(round((x_5.iloc[len(substrate0)
                                                               + len(substrate6_ute)+k][1]-x_5.iloc[len(substrate0)
                                                               + len(substrate6_ute)+k][0])/x_5.iloc[len(substrate0)
                                                               + len(substrate6_ute)+k][0], 2))],
                                    textposition="top center", mode="markers+lines+text"
                                    ),
                         row=k + 1, col=1)
        for v in fig['layout']['annotations']:
            v['font'] = dict(size=9)
    fig.update_xaxes(title_text="T-peel [N]", row=len(substrate0), col=1)
    fig.update_yaxes(title_text="No - number of specimens considered in the series", row=int((len(substrate0) + 1) / 2),
                     col=1)
    for u in range(len(substrate0)):
        fig.update_yaxes(range=[0, 6], row=u + 1, col=1)
    fig.update_layout(height=700, width=500, title_text=tape_1)
    return fig
#################
@app.callback(
    Output('our_graph1', 'figure'),
    [Input('tape_2', 'value')
     ]
)
def build_graph2(tape_2):
    dff = df[df['name'] == tape_2]
    substrate0 = dff[dff['age'] == 0][['substrate']]
    substrate6_ute = dff[(dff['age'] == 6) & (dff['envir'] == 'ute')][['substrate']]
    substrate6_inne = dff[(dff['age'] == 6) & (dff['envir'] == 'inne')][['substrate']]
    dff = dff.sort_values(by=['envir'])
    ###
    x_2 = dff[['TP_stmean2_max', 'TP_stmean2_min', 'TP_stmean2_medi', 'TP_stdev2_min', 'TP_stdev2_max']]
    x_3 = dff[['TP_stmean3_max', 'TP_stmean3_min', 'TP_stmean3_medi', 'TP_stdev3_min', 'TP_stdev3_max']]
    x_4 = dff[['TP_stmean4_max', 'TP_stmean4_min', 'TP_stmean4_medi', 'TP_stdev4_min', 'TP_stdev4_max']]
    x_5_limax = dff['T-peel'] + dff['T-peel stdv']
    x_5_limin = dff['T-peel'] - dff['T-peel stdv']
    x_5 = dff[['T-peel']]
    x_5['T-peel_max'] = x_5_limax
    x_5['T-peel_min'] = x_5_limin
    ###

    fig2 = plt.subplots.make_subplots(rows=len(substrate0), cols=1,
                        subplot_titles=(list(substrate0['substrate'])),
                        vertical_spacing=0.02,
                        shared_xaxes=True)
    for k in range(len(substrate0)):
        if dff['No'].iloc[k] == 2:
            fig2.append_trace(go.Scatter(x=x_2.iloc[k], y=[2] * 5, line=dict(color='red', width=1), name='N = 2'),
                          row =k + 1, col=1)
        else:
            fig2.append_trace(
                go.Scatter(x=x_2.iloc[k], y=[2] * 5, line=dict(color='red', width=1), name='N = 2 (down-sampled)'),
                row=k + 1, col=1)
            fig2.append_trace(go.Scatter(x=x_3.iloc[k], y=[3] * 5, line=dict(color='royalblue', width=1),
                                     name='N = 3 (down-sampled)'),
                row=k + 1, col=1)
            fig2.append_trace(
                go.Scatter(x=x_4.iloc[k], y=[4] * 5, line=dict(color='green', width=1), name='N = 4 (down-sampled)'),
                row=k + 1, col=1)
            fig2.append_trace(go.Scatter(x=x_5.iloc[k], y=[5] * 3, line=dict(color='yellow', width=1), name='N = 5'),
                          row=k + 1, col=1)
    for j in range(len(substrate6_ute)):
        fig2.append_trace(go.Scatter(x=x_5.iloc[len(substrate0) + j], y=[dff['No'].iloc[k]+0.1] * 5,
                                line=dict(color='black', width=1),
                                name ='aged ute: N ='+str(dff['No'].iloc[k])),
                                row =j + 1, col=1)
    for k in range(len(substrate6_inne)):
        fig2.append_trace(go.Scatter(x=x_5.iloc[len(substrate0) + len(substrate6_ute) + k], y=[dff['No'].iloc[k]+0.2] * 5,
                                 line =dict(color='gray', width=1),
                                 name='aged inne: N = '+str(dff['No'].iloc[k])),
                      row = k + 1, col=1)
        for v in fig2['layout']['annotations']:
            v['font'] = dict(size=9)
    fig2.update_xaxes(title_text="T-peel [N]", row=len(substrate0), col=1)
    fig2.update_yaxes(title_text="No - number of specimens considered in the series", row=int((len(substrate0) + 1) / 2),
                     col=1)
    for u in range(len(substrate0)):
        fig2.update_yaxes(range=[1.5, 5.5], row=u + 1, col=1)
    fig2.update_layout(height=700, width=500, title_text=tape_2)
    return fig2
#################
@app.callback(
    Output('our_graph2', 'figure'),
    [Input('tape_3', 'value')
     ]
)
def build_graph3(tape_3):
    dff = df[df['name'] == tape_3]
    substrate0 = dff[dff['age'] == 0][['substrate']]
    substrate6_ute = dff[(dff['age'] == 6) & (dff['envir'] == 'ute')][['substrate']]
    substrate6_inne = dff[(dff['age'] == 6) & (dff['envir'] == 'inne')][['substrate']]
    dff = dff.sort_values(by=['envir'])
    ###
    x_2 = dff[['TP_stmean2_max', 'TP_stmean2_min', 'TP_stmean2_medi', 'TP_stdev2_min', 'TP_stdev2_max']]
    x_3 = dff[['TP_stmean3_max', 'TP_stmean3_min', 'TP_stmean3_medi', 'TP_stdev3_min', 'TP_stdev3_max']]
    x_4 = dff[['TP_stmean4_max', 'TP_stmean4_min', 'TP_stmean4_medi', 'TP_stdev4_min', 'TP_stdev4_max']]
    x_5_limax = dff['T-peel'] + dff['T-peel stdv']
    x_5_limin = dff['T-peel'] - dff['T-peel stdv']
    x_5 = dff[['T-peel']]
    x_5['T-peel_max'] = x_5_limax
    x_5['T-peel_min'] = x_5_limin
    ###
    fig3 = plt.subplots.make_subplots(rows=len(substrate0), cols=1,
                        subplot_titles=(list(substrate0['substrate'])),
                        vertical_spacing=0.02,
                        shared_xaxes=True)
    for k in range(len(substrate0)):
        if dff['No'].iloc[k] == 2:
            fig3.append_trace(go.Scatter(x=x_2.iloc[k], y=[2] * 5, line=dict(color='red', width=1), name='N = 2'),
                          row =k + 1, col=1)
        else:
            fig3.append_trace(
                go.Scatter(x=x_2.iloc[k], y=[2] * 5, line=dict(color='red', width=1), name='N = 2 (down-sampled)'),
                row=k + 1, col=1)
            fig3.append_trace(go.Scatter(x=x_3.iloc[k], y=[3] * 5, line=dict(color='royalblue', width=1),
                                     name='N = 3 (down-sampled)'),
                row=k + 1, col=1)
            fig3.append_trace(
                go.Scatter(x=x_4.iloc[k], y=[4] * 5, line=dict(color='green', width=1), name='N = 4 (down-sampled)'),
                row=k + 1, col=1)
            fig3.append_trace(go.Scatter(x=x_5.iloc[k], y=[5] * 3, line=dict(color='yellow', width=1), name='N = 5'),
                          row=k + 1, col=1)
    for j in range(len(substrate6_ute)):
        fig3.append_trace(go.Scatter(x=x_5.iloc[len(substrate0) + j], y=[dff['No'].iloc[k]+0.1] * 5,
                                line=dict(color='black', width=1),
                                name ='aged ute: N = '+str(dff['No'].iloc[k])),
                                row =j + 1, col=1)
    for k in range(len(substrate6_inne)):
        fig3.append_trace(go.Scatter(x=x_5.iloc[len(substrate0) + len(substrate6_ute) + k], y=[dff['No'].iloc[k]+0.2] * 5,
                                 line =dict(color='gray', width=1),
                                 name='aged inne: N = '+str(dff['No'].iloc[k])),
                      row = k + 1, col=1)
        for v in fig3['layout']['annotations']:
            v['font'] = dict(size=9)
    fig3.update_xaxes(title_text="T-peel [N]", row=len(substrate0), col=1)
    fig3.update_yaxes(title_text="No - number of specimens considered in the series", row=int((len(substrate0) + 1) / 2),
                     col=1)
    for u in range(len(substrate0)):
        fig3.update_yaxes(range=[1.5, 5.5], row=u + 1, col=1)
    fig3.update_layout(height=700, width=500, title_text=tape_3)
    return fig3

