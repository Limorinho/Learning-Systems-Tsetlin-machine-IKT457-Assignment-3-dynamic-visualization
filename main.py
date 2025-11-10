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
    Calculate the stationary distribution for the 8-state Literal Automaton using the exact equations.
    
    From the assignment image, the equations are:
    π₁ = α P(Y)⁴ P(L|Y)⁷
    π₂ = α P(Y)³ P(L|Y)⁶ s (P(L|Y)P(Y) + P(L|Ȳ)P(Ȳ))
    π₃ = α P(Y)² P(L|Y)⁵ s² (P(L|Y)P(Y) + P(L|Ȳ)P(Ȳ))²
    π₄ = α P(Y) P(L|Y)⁴ s³ (P(L|Y)P(Y) + P(L|Ȳ)P(Ȳ))³
    π₅ = α P(L|Y)³ s⁴ (P(L|Y)P(Y) + P(L|Ȳ)P(Ȳ))⁴
    π₆ = α P(L|Y)³ s⁵ (P(L|Y)P(Y) + P(L|Ȳ)P(Ȳ))¹
    π₇ = α P(L|Y)³ s⁶
    π₈ = α P(L|Y)³ s⁷
    """
    # Derived probabilities
    p_l_bar_given_y = 1 - p_l_given_y
    p_y_bar = 1 - p_y
    p_l_given_y_bar = 1 - p_l_bar_given_y_bar
    
    # Calculate the base expression (P(L|Y)P(Y) + P(L|Ȳ)P(Ȳ))
    base_expr = p_l_given_y * p_y + p_l_given_y_bar * p_y_bar
    
    # Calculate each unnormalized probability according to the exact equations
    pi_1_unnorm = p_y**4 * p_l_given_y**7
    pi_2_unnorm = p_y**3 * p_l_given_y**6 * s * base_expr
    pi_3_unnorm = p_y**2 * p_l_given_y**5 * s**2 * base_expr**2
    pi_4_unnorm = p_y * p_l_given_y**4 * s**3 * base_expr**3
    pi_5_unnorm = p_l_given_y**3 * s**4 * base_expr**4
    pi_6_unnorm = p_l_given_y**3 * s**5 * base_expr
    pi_7_unnorm = p_l_given_y**3 * s**6
    pi_8_unnorm = p_l_given_y**3 * s**7
    
    # Calculate normalization constant α
    total_unnorm = (pi_1_unnorm + pi_2_unnorm + pi_3_unnorm + pi_4_unnorm +
                   pi_5_unnorm + pi_6_unnorm + pi_7_unnorm + pi_8_unnorm)
    
    alpha = 1.0 / total_unnorm if total_unnorm > 0 else 1.0
    
    # Calculate normalized probabilities
    stationary_dist = alpha * np.array([pi_1_unnorm, pi_2_unnorm, pi_3_unnorm, pi_4_unnorm,
                                       pi_5_unnorm, pi_6_unnorm, pi_7_unnorm, pi_8_unnorm])
    
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