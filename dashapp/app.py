import dash
import dash_html_components as html

# Initialise the app
app = dash.Dash(__name__)

# Define the app
app.layout = html.Div()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

app.layout = html.Div(children=[
                      html.Div(className='row',  # Define the row element
                               children=[
                                  html.Div(className='four columns div-user-controls'),  # Define the left element
                                  html.Div(className='eight columns div-for-charts bg-grey')  # Define the right element
                                  ])
                                ])

children = [
    html.H2('Dash - STOCK PRICES'),
    html.P('''Visualising time series with Plotly - Dash'''),
    html.P('''Pick one or more stocks from the dropdown below.''')
]