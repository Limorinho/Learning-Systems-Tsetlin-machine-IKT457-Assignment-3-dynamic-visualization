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
                value=0.79,  # Updated to match image
                marks={i/10: str(i/10) for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'marginBottom': '20px'}),
        
        html.Div([
            html.Label("P(L̄|Ȳ) (Probability of not L given not Y):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='p-l-bar-given-y-bar-slider',
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.22,  # Updated so P(L|not Y) = 0.78
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
            "This visualization shows the stationary distribution of a Literal Automaton modeled as a Markov chain with 8 states. "
            "Each state represents a combination of three binary variables: Y (stimulus), L (literal response), and Y_prev (previous stimulus). "
            "The Markov chain captures the temporal dependencies and memory effects in the learning process."
        ]),
        html.P([
            "The stationary distribution π is computed by finding the left eigenvector of the transition matrix P "
            "corresponding to eigenvalue 1, satisfying π = πP. This represents the long-run probability distribution "
            "over states as the system reaches equilibrium."
        ]),
        html.P([
            "Parameters:",
            html.Ul([
                html.Li("s: Strength parameter affecting memory and state transitions"),
                html.Li("P(L|Y): Probability of literal response given Y is true"),
                html.Li("P(Y): Base probability that stimulus Y occurs"),
                html.Li("P(L̄|Ȳ): Probability of not giving literal response when Y is false"),
                html.Li("Transition probabilities depend on current state and parameter values"),
                html.Li("States 1-6 typically represent 'forgotten' configurations with low stationary probability"),
                html.Li("States 7-8 represent 'memorized' configurations with higher stationary probability")
            ])
        ])
    ], style={'marginTop': '30px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'})
])

def calculate_stationary_distribution(s, p_l_given_y, p_y, p_l_bar_given_y_bar):
    """
    Calculate the stationary distribution for the 8-state Literal Automaton using Markov Chain theory.
    
    States represent combinations of (Y, L, Y_prev):
    State 1: (0,0,0) - Y=False, L=False, Y_prev=False  
    State 2: (0,0,1) - Y=False, L=False, Y_prev=True
    State 3: (0,1,0) - Y=False, L=True, Y_prev=False
    State 4: (0,1,1) - Y=False, L=True, Y_prev=True
    State 5: (1,0,0) - Y=True, L=False, Y_prev=False
    State 6: (1,0,1) - Y=True, L=False, Y_prev=True
    State 7: (1,1,0) - Y=True, L=True, Y_prev=False
    State 8: (1,1,1) - Y=True, L=True, Y_prev=True
    """
    # Derived probabilities
    p_l_bar_given_y = 1 - p_l_given_y
    p_y_bar = 1 - p_y
    p_l_given_y_bar = 1 - p_l_bar_given_y_bar
    
    # Create 8x8 transition matrix
    P = np.zeros((8, 8))
    
    # For each current state, calculate transition probabilities to next states
    # Next state depends on: new Y (random), new L (depends on new Y), Y_prev = current Y
    
    for state in range(8):
        # Decode current state (Y, L, Y_prev)
        y_curr = (state // 4) % 2
        l_curr = (state // 2) % 2  
        y_prev_curr = state % 2
        
        # For next step: Y_prev becomes current Y, and we sample new Y and L
        for next_state in range(8):
            # Decode next state
            y_next = (next_state // 4) % 2
            l_next = (next_state // 2) % 2
            y_prev_next = next_state % 2
            
            # Transition is valid only if Y_prev_next == Y_curr
            if y_prev_next == y_curr:
                # Probability of transitioning = P(Y_next) * P(L_next | Y_next)
                prob_y_next = p_y if y_next == 1 else p_y_bar
                
                if y_next == 1:  # Y is true
                    prob_l_next = p_l_given_y if l_next == 1 else p_l_bar_given_y
                else:  # Y is false  
                    prob_l_next = p_l_given_y_bar if l_next == 1 else p_l_bar_given_y_bar
                
                # Apply strength parameter as memory effect
                memory_factor = np.exp(-s * abs(l_curr - l_next))  # Higher s = more memory
                prob_l_next *= memory_factor
                
                P[state, next_state] = prob_y_next * prob_l_next
    
    # Normalize rows to ensure they sum to 1
    for i in range(8):
        row_sum = np.sum(P[i, :])
        if row_sum > 0:
            P[i, :] /= row_sum
    
    # Find stationary distribution by solving π = πP
    # This is equivalent to finding the left eigenvector with eigenvalue 1
    try:
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        
        # Find the eigenvector corresponding to eigenvalue 1
        stationary_idx = np.argmin(np.abs(eigenvalues - 1))
        stationary_dist = np.real(eigenvectors[:, stationary_idx])
        
        # Ensure non-negative and normalize
        stationary_dist = np.abs(stationary_dist)
        stationary_dist = stationary_dist / np.sum(stationary_dist)
        
    except:
        # Fallback: use iterative method
        stationary_dist = np.ones(8) / 8  # Start with uniform distribution
        for _ in range(1000):  # Iterate until convergence
            stationary_dist = stationary_dist @ P
        
    return stationary_dist

@app.callback(
    [Output('stationary-distribution-chart', 'figure'),
     Output('derived-params', 'children')],
    [Input('s-slider', 'value'),
     Input('p-l-given-y-slider', 'value'),
     Input('p-y-slider', 'value'),
     Input('p-l-bar-given-y-bar-slider', 'value')]
)
def update_chart(s, p_l_given_y, p_y, p_l_bar_given_y_bar):
    # Calculate derived parameters
    p_l_bar_given_y = 1 - p_l_given_y
    p_y_bar = 1 - p_y
    p_l_given_y_bar = 1 - p_l_bar_given_y_bar
    
    # Calculate stationary distribution
    stationary_dist = calculate_stationary_distribution(s, p_l_given_y, p_y, p_l_bar_given_y_bar)
    
    # Create state labels (numbered 1-8 to match the image)
    state_labels = [f"State {i+1}" for i in range(8)]
    
    # Create annotations for forgotten/memorized regions
    annotations = []
    # Add "Forgotten" annotation
    annotations.append(dict(
        x=2.5, y=max(stationary_dist[:6]) + 0.02,
        text="Forgotten", showarrow=False,
        font=dict(size=12, color="gray")
    ))
    # Add "Memorized" annotation  
    annotations.append(dict(
        x=6.5, y=max(stationary_dist[6:]) + 0.05,
        text="Memorized", showarrow=False,
        font=dict(size=12, color="gray")
    ))
    
    # Create the bar chart with purple/blue colors to match the image
    colors = ['lightblue'] * 6 + ['mediumpurple'] * 2  # Light blue for forgotten, purple for memorized
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(1, 9)),  # States numbered 1-8
            y=stationary_dist,
            text=[f"{prob:.2f}" if prob > 0.05 else "" for prob in stationary_dist],
            textposition='outside',
            marker_color=colors,
            hovertemplate='<b>State %{x}</b><br>' + 
                         'Probability: %{y:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Interactive Stationary Distribution Chart',
        xaxis_title='',
        yaxis_title='',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 9)),
            ticktext=[str(i) for i in range(1, 9)]
        ),
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False,
        annotations=annotations,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update derived parameters display to match the image format
    derived_params = html.Div([
        html.P(f"P(Y) = {p_y:.2f}", style={'margin': '2px 0'}),
        html.P(f"P(L|Y) = {p_l_given_y:.1f}", style={'margin': '2px 0'}),
        html.P(f"P(L|not Y) = {p_l_given_y_bar:.2f}", style={'margin': '2px 0'})
    ])
    
    return fig, derived_params

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)