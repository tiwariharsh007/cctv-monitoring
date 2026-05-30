import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

data = pd.DataFrame({
    "Time": ["2025-04-25 12:00", "2025-04-25 12:10", "2025-04-25 12:20"],
    "People Count": [10, 15, 20],
})

app.layout = html.Div(children=[
    html.H1(children="Surveillance Dashboard"),
    dcc.Graph(id="people-count-graph"),
    dcc.Graph(id="loitering-heatmap"),
    dcc.Interval(
        id='interval-component',
        interval=10 * 1000,  # in milliseconds
        n_intervals=0
    ),
])

@app.callback(
    dash.dependencies.Output("people-count-graph", "figure"),
    [dash.dependencies.Input("interval-component", "n_intervals")]
)
def update_people_count(n):
    fig = px.line(data, x="Time", y="People Count", title="People Count Over Time")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
