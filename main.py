import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Literal Automaton - Stationary Distribution Visualization", 
            style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    html.Div([
        html.Div([
            html.Label("s (Strength parameter):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='s-slider',
                min=1.0,
                max=25.0,
                step=0.1,
                value=5.0,
                marks={i: str(i) for i in range(1, 26, 4)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'marginBottom': '20px'}),
        
        html.Div([
            html.Label("P(L|Y) (Probability of L given Y):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='p-l-given-y-slider',
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.8,
                marks={i/10: str(i/10) for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'marginBottom': '20px'})
    ]),
    
    html.Div([
        html.Div([
            html.Label("P(Y) (Probability of Y):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='p-y-slider',
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.6,
                marks={i/10: str(i/10) for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'marginBottom': '20px'}),
        
        html.Div([
            html.Label("P(~L|~Y) (Probability of not L given not Y):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='p-not-l-given-not-y-slider',
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.7,
                marks={i/10: str(i/10) for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'marginBottom': '20px'})
    ]),
    
    html.Div([
        html.H3("Derived Parameters:", style={'marginTop': '30px'}),
        html.Div(id='derived-params', style={'marginBottom': '20px', 'padding': '10px', 
                                             'backgroundColor': '#f0f0f0', 'borderRadius': '5px'})
    ]),
    
    dcc.Graph(id='stationary-distribution-chart'),
    
    html.Div([
        html.H3("Model Description:", style={'marginTop': '30px'}),
        html.P([
            "This visualization shows the stationary distribution of a Literal Automaton with 8 states. "
            "The eight states represent all possible combinations of three binary variables: Y, L, and their histories. "
            "The height of each bar represents the long-run probability of being in that state."
        ]),
        html.P([
            "Parameters:",
            html.Ul([
                html.Li("s: Strength parameter affecting the learning dynamics"),
                html.Li("P(L|Y): Probability of literal response given Y is true"),
                html.Li("P(Y): Base probability that Y is true"),
                html.Li("P(~L|~Y): Probability of not giving literal response when Y is false"),
                html.Li("P(~L|Y) = 1 - P(L|Y): Automatically calculated"),
                html.Li("P(~Y) = 1 - P(Y): Automatically calculated")
            ])
        ])
    ], style={'marginTop': '30px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'})
])

def calculate_stationary_distribution(s, p_l_given_y, p_y, p_not_l_given_not_y):
    """
    Calculate the stationary distribution for the 8-state Literal Automaton
    States: (Y, L, Y_prev) where each can be 0 or 1
    """
    # Derived probabilities
    p_not_l_given_y = 1 - p_l_given_y
    p_not_y = 1 - p_y
    p_l_given_not_y = 1 - p_not_l_given_not_y
    
    # Create transition matrix (8x8)
    # States are ordered as: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    # Where each tuple represents (Y, L, Y_prev)
    
    transition_matrix = np.zeros((8, 8))
    
    # Define state transitions based on Literal Automaton dynamics
    # This is a simplified model - you may need to adjust based on your specific assignment requirements
    
    for i in range(8):
        # Decode current state
        y_curr = i // 4
        l_curr = (i // 2) % 2
        y_prev = i % 2
        
        for j in range(8):
            # Decode next state
            y_next = j // 4
            l_next = (j // 2) % 2
            y_prev_next = j % 2
            
            # Y_prev in next state should be Y_curr
            if y_prev_next != y_curr:
                continue
                
            # Calculate transition probability
            prob = 0.0
            
            # Probability of Y_next
            if y_next == 1:
                prob_y_next = p_y
            else:
                prob_y_next = p_not_y
            
            # Probability of L_next given Y_next
            if l_next == 1:
                if y_next == 1:
                    prob_l_next = p_l_given_y
                else:
                    prob_l_next = p_l_given_not_y
            else:
                if y_next == 1:
                    prob_l_next = p_not_l_given_y
                else:
                    prob_l_next = p_not_l_given_not_y
            
            # Apply strength parameter (learning effect)
            strength_factor = np.exp(s * (l_curr if y_curr == 1 else (1-l_curr)))
            prob = prob_y_next * prob_l_next * strength_factor
            
            transition_matrix[i][j] = prob
    
    # Normalize rows
    for i in range(8):
        row_sum = np.sum(transition_matrix[i])
        if row_sum > 0:
            transition_matrix[i] = transition_matrix[i] / row_sum
    
    # Find stationary distribution (eigenvector with eigenvalue 1)
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_idx = np.argmax(np.real(eigenvalues))
    stationary_dist = np.real(eigenvectors[:, stationary_idx])
    stationary_dist = stationary_dist / np.sum(stationary_dist)
    
    return np.abs(stationary_dist)

@app.callback(
    [Output('stationary-distribution-chart', 'figure'),
     Output('derived-params', 'children')],
    [Input('s-slider', 'value'),
     Input('p-l-given-y-slider', 'value'),
     Input('p-y-slider', 'value'),
     Input('p-not-l-given-not-y-slider', 'value')]
)
def update_chart(s, p_l_given_y, p_y, p_not_l_given_not_y):
    # Calculate derived parameters
    p_not_l_given_y = 1 - p_l_given_y
    p_not_y = 1 - p_y
    p_l_given_not_y = 1 - p_not_l_given_not_y
    
    # Calculate stationary distribution
    stationary_dist = calculate_stationary_distribution(s, p_l_given_y, p_y, p_not_l_given_not_y)
    
    # Create state labels
    state_labels = [
        "State 0: (Y=0, L=0, Y_prev=0)",
        "State 1: (Y=0, L=0, Y_prev=1)", 
        "State 2: (Y=0, L=1, Y_prev=0)",
        "State 3: (Y=0, L=1, Y_prev=1)",
        "State 4: (Y=1, L=0, Y_prev=0)",
        "State 5: (Y=1, L=0, Y_prev=1)",
        "State 6: (Y=1, L=1, Y_prev=0)",
        "State 7: (Y=1, L=1, Y_prev=1)"
    ]
    
    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(8)),
            y=stationary_dist,
            text=[f"{prob:.3f}" for prob in stationary_dist],
            textposition='auto',
            marker_color='steelblue',
            hovertemplate='<b>%{text}</b><br>' + 
                         'State: %{x}<br>' + 
                         'Probability: %{y:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Stationary Distribution of 8-State Literal Automaton',
        xaxis_title='State',
        yaxis_title='Probability',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(8)),
            ticktext=[f"S{i}" for i in range(8)]
        ),
        height=500,
        showlegend=False
    )
    
    # Update derived parameters display
    derived_params = html.Div([
        html.P(f"P(~L|Y) = 1 - P(L|Y) = {p_not_l_given_y:.3f}"),
        html.P(f"P(~Y) = 1 - P(Y) = {p_not_y:.3f}"),
        html.P(f"P(L|~Y) = 1 - P(~L|~Y) = {p_l_given_not_y:.3f}")
    ])
    
    return fig, derived_params

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)