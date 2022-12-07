import pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table
from dash import dcc
from dash import html
from plotly.subplots import make_subplots

#
#
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
DATA_PATH_raw = PATH.joinpath("../datasets/raw_data").resolve()
#
# Import data
#
df = pd.read_pickle(DATA_PATH.joinpath('tension.pkl'))
#df = df.sort_values(by=['E'])
tape_set = list(np.unique(list(df['tape type'])))
df = df[['test type', 'conditioning time', 'E x 1mm', 'E', #'tape type',
         'thickness', 'K', 'K_stdv', 'R2', 'R2_stdv', 'F max',
         'F max stdv', 'strain max', 'strain stdv', 'strain lin', 'strain lin stdv']
        ]
name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
        'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W']
E = list(df['E'])
df['name'] = name
#
#
#
# App layout
#
#
df_ = df.sort_values(by=['E'])
fig1 = px.bar(df_, x="name", y="E", title="E tensile tape", labels=dict(E="E [MPa]"))
#
fig2 = px.histogram(df_.iloc[0:-1], x="E", labels=dict(E="E [MPa]"))
#
##### figure with tension plots ###############
NO = 23
fig3 = make_subplots(4, 6, shared_yaxes=True,
                     subplot_titles=('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                                     'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                     'Q', 'R', 'S', 'T', 'U', 'V', 'W'))
#colors = ['chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'crimson', 'cyan']
############################
for k in range(0, NO):
    df_raw = pd.read_pickle(DATA_PATH_raw.joinpath('raw_tens_' + str(k)))
    for j in range(len(df_raw)):
        fig3.add_trace(go.Scatter(x=df_raw['Displ'].iloc[j], y=df_raw['Force'].iloc[j],
        line=dict(color='darkgray', width=2), mode='lines', name=name[k]+' E: '+str(round(E[k], 0))),
        row=int(k/6)+1, col=k % 6 + 1)
fig3.update_yaxes(title_text='Force [N]')  #
fig3.update_xaxes(title_text='Displacement [mm]')
fig3.for_each_annotation(lambda a: a.update(text='tape '+a.text + '  E=' + str(float(df[df['name'] == a.text]['E']))+'MPa'))
fig3.update_layout(
    template="plotly_dark",
    showlegend=False,
    #margin=dict(r=10, t=25, b=40, l=60),
    #annotations=[dict(text="TIGHTEN / SINTEF", showarrow=False, xref="paper", yref="paper", x=0, y=0)]
    )
#############################################################
#
# Application LAYOUT
#
#############################################################
layout = html.Div([
    html.Br(),
    html.Label([' Tighten / SINTEF '],
               style={'font-size': 10, 'font-family': 'Verdana', "text-align": "center"}),
    #html.Img(src=app.get_asset_url('scheme.png'), height=300),
    html.Div(children=[
        dcc.Graph(figure=fig1, style={'display': 'inline-block'}),
        dcc.Graph(figure=fig2, style={'display': 'inline-block'}),
    ]),
    #dcc.Graph(figure=fig1),
    #dcc.Graph(figure=fig2),
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
        page_size=len(df),          # number of rows visible per page
        style_cell={                # ensure adequate header width when text is shorter than cell's text
            'minWidth': 50, 'maxWidth': 90, 'width': 120
        },
        style_cell_conditional=[    # align text columns to left. By default they are aligned to right
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in ['substrate', 'tape type', 'test type', 'load procedure']
        ],
        style_data={                # overflow cells' content into multiple lines
            'whiteSpace': 'normal',
            'height': 'auto'
        }
        ),
    html.Br(),
    dcc.Graph(figure=fig3, id='peel-graph', style={'width': '200vh', 'height': '90vh'}),
])
#
#
#


