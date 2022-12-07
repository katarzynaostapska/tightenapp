import pandas as pd
import numpy as np
import statistics as stat
import plotly.express as px
import dash
from dash import dash_table
from dash import dcc
from dash import html
import pathlib
from app import app
from dash.dependencies import Input, Output
import base64
import plotly.graph_objects as go
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
df = df[df['test type'] == 't-peel'][['No', 'name', 'substrate', 'test type', 'load procedure', 'age',
       'conditioning time', 'T-peel', 'T-peel stdv', 'Peel COV', 'Peel avg', 'Peel avg stdev', 'K_ini', 'K_ini_stdv',
       'Peel LR', 'Peak F', 'Peak F stdev', 'min loc', 'min loc stdev', 'RMSE_mean', 'NRMSE_mean', 'envir',
       'max loc', 'max loc stdev']] #'filename'
#
# figure T-peel for substrates
#
df_viz_1 = df[['substrate', 'name', 'T-peel', 'T-peel stdv', 'K_ini', 'age']]
#fig1 = go.Figure()
fig1 = px.scatter(df_viz_1, x="substrate", y="T-peel",
                    error_y="T-peel stdv", template='none',
                    color="name", symbol="age")
fig1.update_layout(yaxis_range=[0, 100])
#fig1.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
#fig.show()
#
# figure T-peel in K-ini
#
df_viz_2 = df[['name', 'T-peel', 'K_ini', 'K_ini_stdv', 'age', 'T-peel stdv', 'substrate']]
fig2 = px.scatter(df_viz_2, x="K_ini", y="T-peel", template='none',
                 #error_x = 'K_ini_stdv',
                 #error_y = "T-peel stdv",
                 color="name", symbol='age', hover_name='substrate')
fig2.update_layout(xaxis_range=[0, 16], yaxis_range=[0, 100])
#
# figure T-peel in NRMSE
#
df_viz_3 = df[['substrate', 'name', 'T-peel', 'NRMSE_mean', 'K_ini_stdv', 'age', 'T-peel stdv']]
fig3 = px.scatter(df_viz_3, x="NRMSE_mean", y="T-peel", template='none',
                 #error_x = 'K_ini_stdv',
                 error_y = "T-peel stdv",
                 color="name", symbol='age',
                 hover_name='substrate',
                 #text='substrate'
                 )
fig3.update_layout(xaxis_range=[0, 0.2], yaxis_range=[0, 100])
#
# App layout
#
#app = dash.Dash(__name__)
#server = app.server
#
#
#
layout = html.Div([
    html.Br(),
    html.Label([' Tighten / SINTEF '],
               style={'font-size': 10, 'font-family': 'Verdana', "text-align": "center"}),
    html.Img(src=app.get_asset_url('scheme.png'), height=300),
    html.Br(),
    html.Br(),
	dash_table.DataTable(
		id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True, "hideable": True}
            if i in df.columns #"test type" or i == "load procedure" or i == "conditioning time" or i == "RMSE_mean" or i == "NRMSE_mean"
            else {"name": i, "id": i, "deletable": True, "selectable": True}
            for i in df.columns
        ],
        data=df.to_dict('records'),  # the contents of the table
        editable=False,              # allow editing of data inside all cells
        filter_action="native",     # allow filtering of data by user ('native') or not ('none')
        sort_action="native",       # enables data to be sorted per-column by user or not ('none')
        sort_mode="single",         # sort across 'multi' or 'single' columns
        column_selectable="multi",  # allow users to select 'multi' or 'single' columns
        row_selectable="multi",     # allow users to select 'multi' or 'single' rows
        row_deletable=True,         # choose if user can delete a row (True) or not (False)
        selected_columns=[],        # ids of columns that user selects
        selected_rows=[],           # indices of rows that user selects
        page_action="native",       # all data is passed to the table up-front or not ('none')
        page_current=0,             # page number that user is on
        page_size=len(df),                # number of rows visible per page
        style_cell={                # ensure adequate header width when text is shorter than cell's text
            'minWidth': 50, 'maxWidth': 90, 'width': 120
        },
        style_cell_conditional=[    # align text columns to left. By default they are aligned to right
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in ['substrate', 'name', 'test type', 'load procedure']
        ],
        style_data={                # overflow cells' content into multiple lines
            'whiteSpace': 'normal',
            'height': 'auto'
        }
        ),
		html.Br(),
    	html.Br(),
    	html.Div(id='bar-container'),
    	html.Div(id='choromap-container')
	])


#if __name__ == '__main__':
#    app.run_server(debug=True)