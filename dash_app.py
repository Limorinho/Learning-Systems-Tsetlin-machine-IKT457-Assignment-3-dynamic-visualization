# Interactive Stationary Distribution Visualization for Literal Automaton

# Import Required Libraries
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def calculate_stationary_distribution(s, p_l_given_y, p_y, p_not_l_given_not_y):
    """
    Calculate the stationary distribution for the 8-state Literal Automaton using the exact equations.
    """
    # Calculate derived probabilities
    p_not_y = 1 - p_y  # P(Ȳ)
    p_l_given_not_y = 1 - p_not_l_given_not_y  # P(L|Ȳ)
    
    # Calculate the base expression (P(L|Y)P(Y) + P(L|Ȳ)P(Ȳ))
    base_expr = p_l_given_y * p_y + p_l_given_not_y * p_not_y
    
    # Calculate each unnormalized probability according to the equations
    pi_1_unnorm = (p_y**4) * (p_l_given_y**7)
    pi_2_unnorm = (p_y**3) * (p_l_given_y**6) * s * base_expr
    pi_3_unnorm = (p_y**2) * (p_l_given_y**5) * (s**2) * (base_expr**2)
    pi_4_unnorm = p_y * (p_l_given_y**4) * (s**3) * (base_expr**3)
    pi_5_unnorm = (p_l_given_y**3) * (s**4) * (base_expr**4)
    pi_6_unnorm = (p_l_given_y**3) * (s**5) * base_expr
    pi_7_unnorm = (p_l_given_y**3) * (s**6)
    pi_8_unnorm = (p_l_given_y**3) * (s**7)
    
    # Calculate normalization constant α
    total_unnorm = (pi_1_unnorm + pi_2_unnorm + pi_3_unnorm + pi_4_unnorm +
                   pi_5_unnorm + pi_6_unnorm + pi_7_unnorm + pi_8_unnorm)
    
    alpha = 1.0 / total_unnorm if total_unnorm > 0 else 1.0
    
    # Calculate normalized probabilities
    distribution = alpha * np.array([pi_1_unnorm, pi_2_unnorm, pi_3_unnorm, pi_4_unnorm,
                                    pi_5_unnorm, pi_6_unnorm, pi_7_unnorm, pi_8_unnorm])
    
    return distribution

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Literal Automaton - Interactive Stationary Distribution Visualization", 
            style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    # Parameter controls
    html.Div([
        html.Div([
            html.Label("s (Strength parameter):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='s-slider',
                min=1.0,
                max=25.0,
                step=0.1,
                value=1.0,  # Start with value from image
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
                value=0.5,  # Start with value from image
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
                value=0.5,  # Start with value from image
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
                value=0.5,  # Start with value from image
                marks={i/10: str(i/10) for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'marginBottom': '20px'})
    ]),
    
    # Derived parameters display
    html.Div([
        html.H3("Derived Parameters:", style={'marginTop': '30px'}),
        html.Div(id='derived-params', style={'marginBottom': '20px', 'padding': '10px', 
                                             'backgroundColor': '#f0f0f0', 'borderRadius': '5px'})
    ]),
    
    # Bar chart
    dcc.Graph(id='stationary-distribution-chart'),
    
    # Model description
    html.Div([
        html.H3("Model Description:"),
        html.P([
            "This visualization shows the stationary distribution of a Literal Automaton with eight states ",
            "using the exact equations from the assignment. Each πᵢ represents the long-term probability ",
            "of being in state i."
        ]),
        html.P([
            "The strength parameter s and probability parameters control the distribution shape. ",
            "Higher s values typically increase the probability mass in states 7-8."
        ])
    ], style={'marginTop': '30px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'})
])

# Callback to update the chart and derived parameters
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
    
    # Calculate the stationary distribution
    distribution = calculate_stationary_distribution(s, p_l_given_y, p_y, p_not_l_given_not_y)
    
    # Create the bar chart
    colors = ['lightblue'] * 4 + ['mediumpurple'] * 4  # First 4 states vs last 4 states
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(1, 9)),
            y=distribution,
            text=[f"{prob:.3f}" if prob > 0.01 else f"{prob:.4f}" for prob in distribution],
            textposition='outside',
            marker_color=colors,
            hovertemplate='<b>π%{x}</b><br>Probability: %{y:.6f}<extra></extra>'
        )
    ])
    
    # Add annotations for "Forgotten" and "Memorized" regions
    annotations = []
    if max(distribution[:4]) > 0:
        annotations.append(dict(
            x=2.5, y=max(distribution[:4]) + max(distribution) * 0.1,
            text="Forgotten", showarrow=False,
            font=dict(size=14, color="gray")
        ))
    if max(distribution[4:]) > 0:
        annotations.append(dict(
            x=6.5, y=max(distribution[4:]) + max(distribution) * 0.1,
            text="Memorized", showarrow=False,
            font=dict(size=14, color="gray")
        ))
    
    fig.update_layout(
        title='Interactive Stationary Distribution Chart',
        xaxis_title='',
        yaxis_title='π',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 9)),
            ticktext=[str(i) for i in range(1, 9)]
        ),
        yaxis=dict(range=[0, max(distribution) * 1.3] if max(distribution) > 0 else [0, 1]),
        height=500,
        showlegend=False,
        annotations=annotations,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Add vertical line at x=4.5 to separate regions
    fig.add_vline(x=4.5, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Create derived parameters display
    base_expr = p_l_given_y * p_y + p_l_given_not_y * p_not_y
    derived_params = html.Div([
        html.P(f"P(~L|Y) = 1 - P(L|Y) = {p_not_l_given_y:.3f}", style={'margin': '2px 0'}),
        html.P(f"P(~Y) = 1 - P(Y) = {p_not_y:.3f}", style={'margin': '2px 0'}),
        html.P(f"P(L|~Y) = 1 - P(~L|~Y) = {p_l_given_not_y:.3f}", style={'margin': '2px 0'}),
        html.P(f"Base expression = P(L|Y)P(Y) + P(L|~Y)P(~Y) = {base_expr:.3f}", style={'margin': '2px 0'})
    ])
    
    return fig, derived_params

# Run the app
if __name__ == '__main__':
    print("Starting the interactive visualization...")
    print("Open your browser and navigate to http://127.0.0.1:8050/")
    
    # Test the calculation
    print("\nTesting with values from the image (s=1, P(L|Y)=0.5, P(Y)=0.5, P(~L|~Y)=0.5):")
    test_dist = calculate_stationary_distribution(1.0, 0.5, 0.5, 0.5)
    print(f"Distribution: {test_dist}")
    print(f"Sum: {np.sum(test_dist):.6f}")
    for i, prob in enumerate(test_dist, 1):
        print(f"π{i} = {prob:.6f}")
    
    print("\nAlso testing with different values (s=5, P(L|Y)=0.8, P(Y)=0.6, P(~L|~Y)=0.7):")
    test_dist2 = calculate_stationary_distribution(5.0, 0.8, 0.6, 0.7)
    print(f"States 7-8: π₇={test_dist2[6]:.4f}, π₈={test_dist2[7]:.4f}")
    
    # Run the Dash app
    app.run(debug=True, host='0.0.0.0', port=8050)