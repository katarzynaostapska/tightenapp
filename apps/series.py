import pathlib
import statistics as stat
import pandas as pd
import plotly.express as px
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from app import app

# -------------------------------------------------------------------------------------
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
                'conditioning time', 'T-peel', 'T-peel stdv', 'Peel COV',
                'Peel avg', 'Peel avg stdev', 'K_ini', 'K_ini_stdv',
                'Peel LR', 'Peak F', 'Peak F stdev', 'min loc',
                'min loc stdev', 'RMSE_mean', 'NRMSE_mean', 'envir',
                'max loc', 'max loc stdev', 'filename']]

# Creating an ID column name gives us more interactive capabilities
df.index.name = 'id'
#print(df.head())

# -------------------------------------------------------------------------------------
# App layout
#
html.Br(),
html.Label([' Tighten / SINTEF '],
           style={'font-size': 10, 'font-family': 'Verdana', "text-align": "center"}),
html.Br(),
layout = html.Div([
    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True, "hideable": True}
            if i in df.columns
            else {"name": i, "id": i, "deletable": True, "selectable": True}
            for i in df.columns
        ],
        data=df.to_dict('records'),  # the contents of the table
        editable=True,              # allow editing of data inside all cells
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
        page_size=6,                # number of rows visible per page
        style_cell={                # ensure adequate header width when text is shorter than cell's text
            'minWidth': 95, 'maxWidth': 95, 'width': 95
        },
        style_cell_conditional=[    # align text columns to left. By default they are aligned to right
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in ['country', 'iso_alpha3']
        ],
        style_data={                # overflow cells' content into multiple lines
            'whiteSpace': 'normal',
            'height': 'auto'
        }
    ),
    html.Br(),
    html.Br(),
    html.Div(id='scatter-container'),

])


# -------------------------------------------------------------------------------------
# Create scatter chart
@app.callback(
    Output(component_id='scatter-container', component_property='children'),
    [Input(component_id='datatable-interactivity', component_property="derived_virtual_data"),
     Input(component_id='datatable-interactivity', component_property='derived_virtual_selected_rows'),
     Input(component_id='datatable-interactivity', component_property='derived_virtual_selected_row_ids'),
     Input(component_id='datatable-interactivity', component_property='selected_rows'),
     Input(component_id='datatable-interactivity', component_property='derived_virtual_indices'),
     Input(component_id='datatable-interactivity', component_property='derived_virtual_row_ids'),
     Input(component_id='datatable-interactivity', component_property='active_cell'),
     Input(component_id='datatable-interactivity', component_property='selected_cells')]
)
def update_scatter(all_rows_data, slctd_row_indices, slct_rows_names, slctd_rows,
               order_of_rows_indices, order_of_rows_names, actv_cell, slctd_cell):
    print('***************************************************************************')
    print('Data across all pages pre or post filtering: {}'.format(all_rows_data))
    print('---------------------------------------------')
    print("Indices of selected rows if part of table after filtering:{}".format(slctd_row_indices))
    print("Names of selected rows if part of table after filtering: {}".format(slct_rows_names))
    print("Indices of selected rows regardless of filtering results: {}".format(slctd_rows))
    print('---------------------------------------------')
    print("Indices of all rows pre or post filtering: {}".format(order_of_rows_indices))
    print("Names of all rows pre or post filtering: {}".format(order_of_rows_names))
    print("---------------------------------------------")
    print("Complete data of active cell: {}".format(actv_cell))
    print("Complete data of all selected cells: {}".format(slctd_cell))

    dff = pd.DataFrame(all_rows_data)

    # used to highlight selected countries on bar chart
    colors = ['#7FDBFF' if i in slctd_row_indices else '#0074D9'
              for i in range(len(dff))]

    if "T-peel" in dff and "K_ini" in dff:
        return [
            dcc.Graph(id='scatter-chart',
                      figure=px.scatter(
                          data_frame=dff,
                          x="T-peel",
                          y='K_ini',
                          color='name',
                          labels={"T-peel": "N of peel resistance"}
                      ).update_layout(showlegend=False, xaxis={'categoryorder': 'total ascending'})
                      .update_traces(marker_color=colors, hovertemplate="<b>%{y}N</b><extra></extra>")
                      )
        ]

# -------------------------------------------------------------------------------------
# Highlight selected column
@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    [Input('datatable-interactivity', 'selected_columns')]
)
def update_styles(selected_columns):
    return [{
        'if': {'column_id': i},
        'background_color': '#D2F3FF'
    } for i in selected_columns]