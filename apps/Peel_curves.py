import pathlib
import statistics as stat
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from app import app

#
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
DATA_PATH_raw = PATH.joinpath("../datasets/raw_data").resolve()
#
# Import data
#
df = pd.read_pickle(DATA_PATH.joinpath('peel_all_coded'))
#
#
#
#####
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
       'Peel LR', 'Peak F', 'Peak F stdev', 'min loc', 'min loc stdev', 'RMSE', 'RMSE_mean', 'NRMSE', 'NRMSE_mean', 'envir', 'poly_c',
       'max loc', 'max loc stdev', 'filename', 'E', 'Density', 'Surf Energy', 'Displ 0', 'Displ end', 'Displ 0.25', 'Displ 0.75', 'Displ ini']]
########
tape_set = list(np.unique(list(df['name'])))
print(tape_set)
substrate_set = list(np.unique(list(df['substrate'])))
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
    html.Div(children=[
        html.Label(['Results of experimental peel test in the form of force-displacement graph. '
                    'For a given tape and substrate curves for all specimens in a series are shown. '
                    'Fresh specimens, specimens aged outside and inside are shown on separate graphs. '
                    'A 10th order polynomial fitting is depicted together with other statistic parameters. '],
                    style={'font-size': 16, 'font-family': 'Verdana',
                           "text-align": "left", 'display': 'inline-block'}),
        html.Img(src=app.get_asset_url('scheme.png'), style={'display': 'inline-block', 'height': '300px'}),
        ]),
    html.Br(),
    html.Label([' Select a tape: '
                ],
               style={'font-size': 12, 'font-family': 'Verdana', "text-align": "center"}),
    html.Br(),
    dcc.Dropdown(id='tapes',
                 options=[{'label': x, 'value': x} for x in
                          tape_set],
                 value='A',
                 multi=False,
                 disabled=False,
                 clearable=True,
                 searchable=True,
                 placeholder='Choose tape...',
                 className='form-dropdown',
                 style={'width': "100%", 'backgroundColor': 'lightgray'},
                 persistence='string',
                 persistence_type='memory'),
    html.Br(),
    html.Label([' Select a substrate '
                ],
               style={'font-size': 12, 'font-family': 'Verdana', "text-align": "center"}),
    dcc.RadioItems(id='substrates',
                 options=[],
                 value=[],
                 ),
    html.Br(),
    html.Br(),
    html.Div(children=[
        html.Label(id='title1', style={'font-weight': 'bold', 'display': 'inline-block',
                                       "margin-left": "10px", "margin-right": "10px"}),
             ], style={'backgroundColor': 'lightgray', 'height': '5vh'}),
    dcc.Graph(id='peel-graph', style={'width': '200vh', 'height': '90vh'}),
    html.Br(style={'backgroundColor': 'lightgray', 'height': '90vh'}),
])
# -------------------------------------------------------------------------------------
# Populate the substrate dropdown with options and values
@app.callback(
    Output('substrates', 'options'),
    Output('substrates', 'value'),
    Input('tapes', 'value'),
)
def set_substrate_options(chosen_tape):
    dff = df[df.name == chosen_tape]
    substrates_of_tape = [{'label': c, 'value': c} for c in dff.substrate.unique()]
    values_selected = [x['value'] for x in substrates_of_tape]
    return substrates_of_tape, values_selected
# Create scatter chart
@app.callback(
    Output('peel-graph', 'figure'),
    Output('title1', 'children'),
    [Input('tapes', 'value'),
     Input('substrates', 'value')
     ]
)
def build_graph(tape_set, substrate_set):
    #
    name = tape_set
    substrate = substrate_set
    print(name, substrate)
    df_ = df[(df['name'] == name)]
    df_ = df_[df_['substrate'] == substrate]
    indexes = df_.index
    df_raw_f = pd.read_pickle(DATA_PATH_raw.joinpath('raw_fresh_' + str(indexes[0])))
    df_raw = pd.DataFrame()
    print(df_raw)
    df_list = []
    df_list.append(df_raw_f)
    for j in range(len(indexes)-1):
        print('j', j, indexes[j+1]-115)
        df_raw_a = pd.read_pickle(DATA_PATH_raw.joinpath('raw_aged_' + str(indexes[j+1]-115)))
        df_list.append(df_raw_a)
        df_raw = pd.concat([df_raw, df_raw_a], ignore_index=False)
    #df_raw.reset_index()
    #df_raw.drop_duplicates('Force')
    print('df_list: ', df_list)
    print(df_, indexes, df_raw_a)
    print(df_raw)
    df_raw = pd.concat([df_raw_f, df_raw], ignore_index=False)
    print(df_raw)
    df_raw_all = df_raw.reset_index(drop=True)
    print(df_raw_all)
    x_0 = list(df_[(df_['name'] == name)]['Displ 0'])
    x_end = list(df_[(df_['name'] == name)]['Displ end'])
    x_25 = list(df_[(df_['name'] == name)]['Displ 0.25'])
    x_75 = list(df_[(df_['name'] == name)]['Displ 0.75'])
    x_ini = list(df_[(df_['name'] == name)]['Displ ini'])
    subst = list(df_['substrate'])
    envir = list(df_['envir'])
    peak_F = list(df_['Peak F'])
    print(peak_F)
    age = list(df_['age'])
    y = list(df_['poly_c'])
    print(y)
    plot_names = []
    string = []
    for u in range(0, len(df_)):
        Tpeel = round(df_['T-peel'].iloc[u], 1)
        Tpeel_std = round(df_['T-peel stdv'].iloc[u], 2)
        Kini = round(df_['K_ini'].iloc[u], 2)
        Kini_std = round(df_['K_ini_stdv'].iloc[u], 2)
        plot_names.append([age[u], Kini, Kini_std, Tpeel,
                           Tpeel_std, [round(z, 2) for z in df_['RMSE'].iloc[u]],
                           [round(z, 2) for z in df_['NRMSE'].iloc[u]]])
        string.append('   |  age: ' + str(plot_names[u][0]) + ' ' + str(envir[u])\
                      + ' K-ini: ' + str(plot_names[u][1]) + '±' + str(plot_names[u][2])\
                      + ' T-peel: ' + str(plot_names[u][3]) + '±' + str(plot_names[u][4])\
                      + '\n' + ' NRMSE: ' + str(plot_names[u][6])+'--------')
    ##### figure ###############
    fig = make_subplots(1, len(df_), shared_yaxes=False)
    colors = ['chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'crimson', 'cyan']
    ############################
    for k in range(0, len(df_)):
        Kini = round(df_['K_ini'].iloc[k], 2)
        fig.add_trace(go.Scatter
                      (x=[x_0[k][0], x_end[k][0] + 0.5], y=[df_['min loc'].iloc[k], df_['min loc'].iloc[k]],
                       line=dict(color='darkgray', width=2, dash='dot'), mode='lines', name='min-' + str(envir[k])),
                      row=1, col=k + 1)
        fig.add_trace(go.Scatter
                      (x=[x_0[k][0], x_end[k][0] + 0.5], y=[df_['max loc'].iloc[k], df_['max loc'].iloc[k]],
                       line=dict(color='darkgray', width=2, dash='dot'), mode='lines', name='max-' + str(envir[k])),
                      row=1, col=k + 1)
        fig.add_trace(go.Scatter
                      (x=[x_0[k][0], x_end[k][0] + 0.5], y=[df_['T-peel'].iloc[k], df_['T-peel'].iloc[k]],
                       line=dict(color='lightgray', width=2, dash='dot'), mode='lines', name='peel-' + str(envir[k])),
                      row=1, col=k + 1)
        fig.add_trace(go.Scatter
                      (x=[x_25[k][0], x_25[k][0]], y=[0, df_['T-peel'].iloc[k]+ 15],
                       line=dict(color='lightgray', width=0.5, dash='dash'), mode='lines',
                       name='x_25%'),
                      row=1, col=k + 1)
        fig.add_trace(go.Scatter
                      (x=[x_75[k][0], x_75[k][0]], y=[0, df_['T-peel'].iloc[k] + 15],
                       line=dict(color='lightgray', width=0.5, dash='dash'), mode='lines',
                       name='x_75%'),
                      row=1, col=k + 1)
        for j in range(df_['No'].iloc[k]):
            poly = np.poly1d(y[k][j])
            domain = list(np.arange(x_0[k][j], x_end[k][j]))
            print(j+k*df_['No'].iloc[k], j, k, df_['No'].iloc[k])
            #fig.add_trace(go.Scatter(x=df_raw_all['Displ'][j+k*df_['No'].iloc[k]],
            #                         y=[50/34*z for z in df_raw_all['Force'][j+k*df_['No'].iloc[k]]],
            #                         line=dict(color='gray', width=0.75), mode='lines', name='test-'+str(j)),
            #              row=1, col=k + 1)
            fig.add_trace(go.Scatter(x=df_list[k]['Displ'][j],
                                     y=[50/34*z for z in df_list[k]['Force'][j]],
                                     line=dict(color='gray', width=0.75), mode='lines', name='test-'+str(j)),
                          row=1, col=k + 1)
            fig.add_trace(go.Scatter(x=domain, y=poly(domain),
                          line=dict(color=colors[j], width=2.5), mode='lines', name='poly-'+str(j)),
                          row=1, col=k + 1)
            fig.add_trace(go.Scatter
                          (x=[x_0[k][j], x_ini[k][j]],
                           y=[x_0[k][j], Kini * (x_ini[k][j] - x_0[k][j])],
                           line=dict(color='red', width=2, dash='dot'), mode='lines', name='K-ini-'+str(j)),
                          row=1, col=k + 1)
    fig.update_yaxes(title_text='Force [N]', range=[0, max(peak_F)*1.05])
    fig.update_xaxes(title_text='Displacement [mm]')
    fig.update_layout(template="plotly_dark",
                    margin=dict(r=10, t=25, b=40, l=60),
                    annotations=[dict(text="TIGHTEN / SINTEF",
                    showarrow=False, xref="paper", yref="paper", x=0, y=0)])
    return fig, string

