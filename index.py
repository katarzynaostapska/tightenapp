import dash_auth
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from apps import tighten, tighten2, PCA, Stats, series, Peel_curves, tension
############################
#         validation
###
VALID_USERNAME_PASSWORD_PAIRS = {
    'kasia': 'tighten_kasia',
    'klodian': 'tighten_klodian',
    'malin': 'tighten_malin',
    'petra': 'tighten_petra',
    'susanne': 'tighten_susanne'
}
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
###############################
#       PAGE LAYOUT
###############################
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Link('      Overview ', href='/apps/tighten', style={"color": "gray", "size": "24", 'font-weight': 'bold'}),
        dcc.Link('      Data table', href='/apps/tighten2', style={"color": "gray", "size": "24", 'font-weight': 'bold', "margin-left": "15px"}),
        dcc.Link('      PCA ', href='/apps/PCA', style={"color": "gray", "size": "24", 'font-weight': 'bold', "margin-left": "15px"}),
        dcc.Link('      Statistics ', href='/apps/Stats', style={"color": "gray", "size": "24", 'font-weight': 'bold', "margin-left": "15px"}),
        dcc.Link('      Test Series ', href='/apps/series', style={"color": "gray", "size": "24", 'font-weight': 'bold', "margin-left": "15px"}),
        dcc.Link('      Peel curves ', href='/apps/Peel_curves', style={"color": "gray", "size": "24", 'font-weight': 'bold', "margin-left": "15px"}),
        dcc.Link('      Tension ', href='/apps/tension', style={"color": "gray", "size": "24", 'font-weight': 'bold', "margin-left": "15px"}),
    ], className="row"),
    html.Div(id='page-content', children=[]),
    html.Br(),
    html.Br(),
    html.Img(src=app.get_asset_url('simulation.png'), height=600),
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/tighten':
        return tighten.layout
    if pathname == '/apps/tighten2':
        return tighten2.layout
    if pathname == '/apps/PCA':
        return PCA.layout
    if pathname == '/apps/Stats':
        return Stats.layout
    if pathname == '/apps/series':
        return series.layout
    if pathname == '/apps/Peel_curves':
        return Peel_curves.layout
    if pathname == '/apps/tension':
        return tension.layout
    else:
        return " Please choose a link above "


if __name__ == '__main__':
    app.run_server(debug=False)
