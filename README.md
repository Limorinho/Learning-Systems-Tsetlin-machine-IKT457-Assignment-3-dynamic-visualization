# Literal Automaton Visualization

An interactive visualization of the stationary distribution for an 8-state Literal Automaton. This web application allows you to adjust parameters in real-time and observe how they affect the distribution.

## Live Demo

Visit the live demo: [Your GitHub Pages URL will be here]

## Features

- **Interactive Sliders**: Adjust parameters s, P(L|Y), P(Y), and P(~L|~Y)
- **Real-time Updates**: Chart updates instantly as you change parameters
- **Derived Parameters**: Automatically calculated complementary probabilities
- **Responsive Design**: Works on desktop and mobile devices
- **Mathematical Accuracy**: Implements the full transition matrix calculation

## Parameters

- **s (Strength parameter)**: Range 1-25, affects learning dynamics
- **P(L|Y)**: Probability of literal response given Y is true (0-1)
- **P(Y)**: Base probability that Y is true (0-1)
- **P(~L|~Y)**: Probability of not giving literal response when Y is false (0-1)

## Model Description

The visualization shows the stationary distribution of a Literal Automaton with 8 states representing all possible combinations of three binary variables: Y, L, and their histories. The height of each bar represents the long-run probability of being in that state.

## Technology Stack

- HTML5
- CSS3
- JavaScript (ES6)
- Plotly.js for interactive charts

## Local Development

Simply open `index.html` in any modern web browser to run the visualization locally.

## GitHub Pages Deployment

This project is designed to be deployed on GitHub Pages. The main visualization is in `index.html` which can be served directly as a static website.