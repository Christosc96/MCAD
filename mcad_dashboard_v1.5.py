import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from itertools import cycle, combinations
import json
import random
import stumpy
import math
from stumpy import config
import io
import base64

config.STUMPY_EXCL_ZONE_DENOM = 1

class TimeSeriesGenerator:
    """Handles time series data generation"""
    
    def __init__(self):
        self.template_functions = [
            self.heartbeat_template,
            self.step_recovery_noisy,
            self.on_off_cycle_ringing,
            self.prolonged_on_noisy
        ]
        self.template_names = ["Heartbeat", "Step Recovery", "On-Off Ringing", "Prolonged On"]
    
    def heartbeat_template(self, length, amplitude=1.0):
        t = np.linspace(0, 1, length)
        return amplitude * (
            np.exp(-((t - 0.2)/0.03)**2) * 0.2 +
            np.exp(-((t - 0.4)/0.02)**2) * -0.3 +
            np.exp(-((t - 0.45)/0.01)**2) * 1.5 +
            np.exp(-((t - 0.5)/0.02)**2) * -0.4 +
            np.exp(-((t - 0.7)/0.06)**2) * 0.3
        )

    def step_recovery_noisy(self, length, amplitude=1.0, noise_std=0.2, seed=None):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, length)
        step = (t > 0.2).astype(float)
        recovery = np.exp(-6 * (t - 0.2)) * (t > 0.2)
        clean = amplitude * step * recovery
        noise = rng.normal(0, noise_std, length)
        return clean + noise

    def on_off_cycle_ringing(self, length, amplitude=2):
        t = np.linspace(0, 1, length)
        rise = 1 / (1 + np.exp(-12 * (t - 0.2)))
        fall = 1 / (1 + np.exp(-12 * (t - 0.6)))
        base = amplitude * (rise - fall)
        ringing = 0.12 * np.sin(18 * np.pi * t) * np.exp(-10 * (t - 0.2))
        ringing *= (t > 0.2)
        return base + ringing

    def prolonged_on_noisy(self, length, amplitude=1.0, noise_std=0.2, seed=None):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, length)
        rise = 1 / (1 + np.exp(-10 * (t - 0.15)))
        noise = np.random.normal(0, 0.01, length)
        return amplitude * rise + noise
    
    def generate_data(self, T=5000, N=4, k=3, random_templates=True, discord_length=50, normality_coef = 2):
        """Generate time series data with events"""
        np.random.seed(np.random.randint(0, 10000))
        
        event_lengths = [discord_length] * N
        min_gaps = [50] * N
        
        # Check for warning condition
        warning_message = None
        normal_series_length = 0  # Will be calculated based on actual events
        if (T - discord_length + min_gaps[0]) < T:  # Simplified check
            # More detailed check will be done after event generation
            pass
        
        # Generate or assign templates
        if random_templates:
            rng_template = np.random.default_rng()
            channel_template_indices = rng_template.choice(len(self.template_functions), 
                                                           size=N, replace=True)
        else:
            channel_template_indices = [0] * N
        
        # Generate templates for each channel
        templates = []
        for i in range(N):
            template_func = self.template_functions[channel_template_indices[i]]
            templates.append(template_func(event_lengths[i], amplitude=2.0))
        
        min_gaps = [50] * N
        
        # Background
        series = np.random.normal(0, 0.01, (N, T))
        
        # Track all events
        events_log = []
        
        subsets = []
        for k_c in range(2, k):
            subsets = subsets + list(combinations(range(N), k_c))

        random.shuffle(subsets)

        
        # Choose k-series for unique event
        rng = np.random.default_rng()
        k_series = rng.choice(N, size=k, replace=False)
        
        # Forced k-way event timing
        forced_t = T//2
        k_way_start = forced_t
        k_way_end = forced_t + max(event_lengths[i] for i in k_series)
        
        # Generate normal events
        for t, subset in zip(range(0, T - event_lengths[0] + min_gaps[0], 
                                   (min_gaps[0] + event_lengths[0])), cycle(subsets)):                                  

            if k_way_start <= t < k_way_end:
                continue
            
            max_end = t
            for channel_idx in subset:
                L = event_lengths[channel_idx]
                end = min(T, t + L)
                series[channel_idx, t:end] += templates[channel_idx][:end - t]
                max_end = max(max_end, end)
            
            events_log.append((t, max_end, frozenset(subset)))
        
        # Insert unique k-way event
        max_end = forced_t
        for i in k_series:
            L = event_lengths[i]
            end = forced_t + L
            series[i, forced_t:end] += templates[i]
            max_end = max(max_end, end)
        
        events_log.append((forced_t, max_end, frozenset(k_series)))
        
        # Calculate warning condition based on the original logic
        # Count normal events (excluding the k-way anomaly)
        normal_events = [e for e in events_log if frozenset(k_series) != e[2]]
        normal_series_length = len(normal_events) * (discord_length + min_gaps[0])
        
        # Check if series is too short for optimal event distribution
        available_space = T - discord_length + min_gaps[0]
        if available_space < normality_coef*len(subsets)*(discord_length + min_gaps[0]):
            warning_message = (
                f"⚠️ Warning: Series too short for optimal event distribution! "
                f"Available space ({available_space}) is less than "
                f"required space for normal events pattern ({normality_coef*len(subsets)*(discord_length + min_gaps[0])}). "
                f"Consider increasing Series Length or decreasing Channels/Arity."
            )
        
        return {
            'series': series,
            'events_log': events_log,
            'k_way_start': k_way_start,
            'k_way_end': k_way_end,
            'k_series': list(k_series),
            'channel_template_indices': list(channel_template_indices),
            'template_names': [self.template_names[i] for i in channel_template_indices],
            'warning': warning_message
        }


# Discord Profile Analysis Functions
def flatten_sliding_windows(X, W, stride=1, normalize = False):
    """Flatten sliding windows for multivariate time series"""
    d, N = X.shape

    # Shape: (d, num_windows_full, W)
    windows = np.lib.stride_tricks.sliding_window_view(X, window_shape=W, axis=1)

    # Apply stride by slicing along the windows axis
    windows = windows[:, ::stride, :]  # (d, num_windows, W)

    num_windows = windows.shape[1]

    # Rearrange to (num_windows, d, W) then flatten
    return windows.transpose(1, 0, 2).reshape(num_windows, d * W).flatten()


def subsets(S):
    """Generate all subsets of a set S"""
    result = []
    for r in range(len(S)+1):
        for c in combinations(S, r):
            result.append(set(c))
    return result


def supersets(S, n):
    """Generate all supersets of S within universe of size n"""
    S = set(S)
    U = set(range(n))
    remaining = list(U - S)

    result = []
    for r in range(len(remaining)+1):
        for c in combinations(remaining, r):
            result.append(S | set(c))
    return result


def discord_profile(S, n_channels):
    """Generate discord profile: all subsets and supersets of S"""
    discord_prof = subsets(S) + supersets(S, n_channels)
    discord_prof = [list(s) for s in discord_prof]
    return discord_prof


def get_discord_score(X, subset=None, m=50):
    """Calculate discord score for a channel subset"""
    if subset is not None and len(subset) > 0:
        X_sub = X[subset, :]
    else:
        X_sub = X
    
    new_m = m * X_sub.shape[0]
    
    # Flatten sliding windows
    flattened = flatten_sliding_windows(X_sub, m, m)
    
    if flattened.shape[0] < 2:
        return 0.0
    
    # Calculate matrix profile
    matrix_profile = stumpy.stump(flattened, m=new_m, normalize=False)
    
    # Get top discord
    top_k_idx = np.argsort(matrix_profile[:, 0] * math.sqrt(1/new_m))[-1]
    top_k_dists = matrix_profile[top_k_idx, 0] * math.sqrt(1/new_m)
    
    return top_k_dists


# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
generator = TimeSeriesGenerator()

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
            }
            
            .dashboard-container {
                display: flex;
                height: 100vh;
                overflow: hidden;
            }
            
            .control-panel {
                width: 280px;
                background-color: #ffffff;
                border-right: 1.5px solid #e1e4e8;
                padding: 1.5rem;
                overflow-y: auto;
                box-shadow: 2px 0 8px rgba(0,0,0,0.05);
            }
            
            .control-panel-header {
                text-align: center;
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 1.5px solid #dee2e6;
            }
            
            .control-panel-icon {
                font-size: 24px;
                color: #0366d6;
                margin-bottom: 0.5rem;
            }
            
            .control-panel-title {
                font-size: 15px;
                font-weight: 600;
                color: #24292e;
                margin: 0;
                letter-spacing: 0.3px;
            }
            
            .section-header {
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                color: #6a737d;
                margin-top: 1.25rem;
                margin-bottom: 0.75rem;
                letter-spacing: 0.5px;
            }
            
            .control-group {
                margin-bottom: 1.25rem;
            }
            
            .control-label {
                display: block;
                font-size: 13px;
                font-weight: 500;
                color: #24292e;
                margin-bottom: 0.4rem;
            }
            
            .control-input {
                width: 100%;
                padding: 0.5rem 0.75rem;
                border: 1px solid #d1d5da;
                border-radius: 6px;
                font-size: 13px;
                background-color: #fafbfc;
                transition: all 0.2s;
            }
            
            .control-input:focus {
                outline: none;
                border-color: #0366d6;
                background-color: #ffffff;
                box-shadow: 0 0 0 3px rgba(3, 102, 214, 0.1);
            }
            
            .btn-primary {
                width: 100%;
                padding: 0.65rem 1rem;
                background: linear-gradient(180deg, #2ea44f 0%, #22863a 100%);
                color: white;
                border: 1px solid rgba(27, 31, 35, 0.15);
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.15s;
                box-shadow: 0 1px 0 rgba(27, 31, 35, 0.04);
            }
            
            .btn-primary:hover {
                background: linear-gradient(180deg, #2c974b 0%, #1f7f34 100%);
                box-shadow: 0 1px 0 rgba(27, 31, 35, 0.1);
            }
            
            .btn-primary:active {
                background: #22863a;
                box-shadow: inset 0 1px 0 rgba(20, 70, 32, 0.2);
            }
            
            .btn-secondary {
                width: 100%;
                padding: 0.65rem 1rem;
                background: linear-gradient(180deg, #fafbfc 0%, #e1e4e8 100%);
                color: #24292e;
                border: 1px solid rgba(27, 31, 35, 0.15);
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.15s;
                margin-top: 0.5rem;
            }
            
            .btn-secondary:hover {
                background: linear-gradient(180deg, #f3f4f6 0%, #d1d5da 100%);
            }
            
            .main-content {
                flex: 1;
                overflow-y: auto;
                padding: 1.5rem;
                background-color: #f8f9fa;
            }
            
            .warning-banner {
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 6px;
                padding: 0.75rem 1rem;
                margin-bottom: 1rem;
                font-size: 13px;
                color: #856404;
            }
            
            .card-compact {
                background: white;
                border-radius: 18px;
                box-shadow: 0 1px 3px rgba(27, 31, 35, 0.12);
                border: 1px solid #e1e4e8;
            }
            
            .card-body {
                padding: 0.75rem;
            }
            
            .card-body h5 {
                font-size: 13px;
                font-weight: 600;
                color: #24292e;
                margin: 0 0 0.5rem 0;
            }
            
            
            .hover-info {
                position: fixed !important;
                bottom: 1rem;
                left: 50% !important;
                right: auto !important;

                background: rgba(36, 41, 46, 0.95);
                color: white;
                padding: 0.75rem 1.25rem;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                opacity: 0;
                transition: opacity 0.2s;
                pointer-events: none;
                z-index: 1000;
            }
            
            .hover-info.active {
                opacity: 1;
            }
            
            .checkbox-group {
                margin: 0.5rem 0;
            }
            
            .checkbox-item {
                display: flex;
                align-items: center;
                margin-bottom: 0.5rem;
                padding: 0.4rem;
                border-radius: 4px;
                transition: background-color 0.2s;
            }
            
            .checkbox-item:hover {
                background-color: #f6f8fa;
            }
            
            .checkbox-item input[type="checkbox"] {
                margin-right: 0.5rem;
                cursor: pointer;
            }
            
            .checkbox-item label {
                font-size: 13px;
                color: #24292e;
                cursor: pointer;
                flex: 1;
            }
            
            .discord-analysis-section {
                background: white;
                border-radius: 8px;
                padding: 1.5rem;
                margin-top: 1rem;
                box-shadow: 0 1px 3px rgba(27, 31, 35, 0.12);
                border: 1px solid #e1e4e8;
            }
            
            .analysis-header {
                font-size: 16px;
                font-weight: 600;
                color: #24292e;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #e1e4e8;
            }
            
            .loading-spinner {
                display: inline-block;
                width: 14px;
                height: 14px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #0366d6;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 0.5rem;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = html.Div([
    dcc.Store(id='data-store'),
    dcc.Download(id='download-npy'),
    
    html.Div([
        # Left control panel
        html.Div([
            html.Div([
                html.H3('MCAD VIZ', className='control-panel-title')
            ], className='control-panel-header'),
            
            html.Div('Data Generation', className='section-header'),
            
            html.Div([
                html.Label('Series Length (T)', className='control-label'),
                dcc.Input(id='T-input', type='number', value=5000, className='control-input')
            ], className='control-group'),
            
            html.Div([
                html.Label('Channels (N)', className='control-label'),
                dcc.Input(id='N-input', type='number', value=4, min=2, max=8, className='control-input')
            ], className='control-group'),
            
            html.Div([
                html.Label('Anomaly Arity (k)', className='control-label'),
                dcc.Input(id='k-input', type='number', value=3, min=2, className='control-input')
            ], className='control-group'),
            
            html.Div([
                html.Label('Event Length (m)', className='control-label'),
                dcc.Input(id='discord-length-input', type='number', value=50, min=10, className='control-input')
            ], className='control-group'),
            
            html.Div([
                html.Label('Normality Coefficient', className='control-label'),
                dcc.Input(id='normality-coef-input', type='number', value=2, min=1, className='control-input')
            ], className='control-group'),
            
            html.Button('Generate Data', id='generate-btn', n_clicks=0, className='btn-primary'),
            html.Button('Download .npy', id='download-npy-btn', n_clicks=0, className='btn-secondary'),

            # Discord Analysis Section
            html.Div('Discord Profile Analysis', className='section-header', style={'marginTop': '2rem'}),
            
            html.Div([
                html.Label('Select Channels for Analysis', className='control-label'),
                html.Div(id='channel-selector', className='checkbox-group')
            ], className='control-group'),
            
            html.Div([
                html.Label('Window Size (m)', className='control-label'),
                dcc.Input(id='analysis-m-input', type='number', value=50, min=10, className='control-input')
            ], className='control-group'),
            
            html.Button('Run Discord Analysis', id='analyze-btn', n_clicks=0, className='btn-secondary'),
            
            html.Div(id='analysis-status', style={'marginTop': '0.5rem', 'fontSize': '12px', 'color': '#6a737d'})
            
        ], className='control-panel'),
        
        # Main content area
        html.Div([
            html.Div(id='warning-container'),
            html.Div(id='plots-container'),
            html.Div(id='discord-results-container'),
            html.Div(id='hover-info', className='hover-info')
        ], className='main-content')
        
    ], className='dashboard-container')
])


# Callback to generate data
@app.callback(
    Output('data-store', 'data'),
    Output('plots-container', 'children'),
    Output('warning-container', 'children'),
    Output('channel-selector', 'children'),
    Input('generate-btn', 'n_clicks'),
    State('T-input', 'value'),
    State('N-input', 'value'),
    State('k-input', 'value'),
    State('discord-length-input', 'value'),
    State('normality-coef-input', 'value'),
    prevent_initial_call=True
)
def generate_data(n_clicks, T, N, k, discord_length, normality_coef):
    result = generator.generate_data(
        T=T, 
        N=N, 
        k=k, 
        random_templates=True, 
        discord_length=discord_length,
        normality_coef=normality_coef
    )
    
    # Store data
    data = {
        'series': result['series'].tolist(),
        'T': T,
        'N': N,
        'k': k,
        'discord_length': discord_length,
        'k_way_start': result['k_way_start'],
        'k_way_end': result['k_way_end'],
        'template_names': result['template_names'],
        'events_log': [(s, e, list(c)) for s, e, c in result['events_log']]
    }
    
    # Generate plots
    plots = create_plots(data)
    
    # Warning banner
    warning = None
    if result['warning']:
        warning = html.Div(result['warning'], className='warning-banner')
    
    # Channel selector checkboxes
    channel_checkboxes = []
    for i in range(N):
        channel_checkboxes.append(
            html.Div([
                dcc.Checklist(
                    id={'type': 'channel-checkbox', 'index': i},
                    options=[{'label': f' Channel {i}', 'value': i}],
                    value=[],
                    style={'margin': 0}
                )
            ], className='checkbox-item')
        )
    
    return data, plots, warning, channel_checkboxes


def create_plots(data):
    """Create time series plots"""
    if data is None:
        return []
    
    series = np.array(data['series'])
    N = data['N']
    T = data['T']
    k_way_start = data['k_way_start']
    k_way_end = data['k_way_end']
    template_names = data['template_names']
    events_log = data['events_log']
    
    plot_colors = ['#0366d6', '#6f42c1', '#d73a49', '#28a745', 
                   '#ffa500', '#e36209', '#0366d6', '#6610f2']
    
    # Calculate dynamic height to fit all plots on screen
    available_height = 85  # vh units
    plot_height = max(120, int((available_height * 10) / N))
    
    plots = []
    
    for i in range(N):
        fig = go.Figure()
        
        # Add the time series line
        fig.add_trace(go.Scatter(
            x=list(range(T)),
            y=series[i],
            mode='lines',
            line=dict(color=plot_colors[i % len(plot_colors)], width=1.2),
            name=f'Channel {i}',
            customdata=[[i, t] for t in range(T)],
            hoverinfo='x+y',
            hovertemplate='Time: %{x}<br>Value: %{y:.3f}<extra></extra>'
        ))
        
        # Add k-way event region as a shape
        fig.add_shape(
            type="rect",
            x0=k_way_start,
            x1=k_way_end,
            y0=series[i].min() - 0.1,
            y1=series[i].max() + 0.1,
            fillcolor="rgba(220, 53, 69, 0.12)",
            line=dict(width=0),
            layer="below",
            name="k_way_region"
        )
        
        # Update layout
        fig.update_layout(
            margin=dict(l=10, r=10, t=5, b=25 if i == N-1 else 15),
            height=plot_height,
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(209, 213, 218, 0.3)',
                gridwidth=0.5,
                zeroline=False,
                title=dict(text='Time' if i == N-1 else '', font=dict(size=10)),
                tickfont=dict(size=8, color='#6a737d'),
                range=[0, T]
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(209, 213, 218, 0.3)',
                gridwidth=0.5,
                zeroline=False,
                title=dict(
                    text=f'Ch {i}<br>{template_names[i][:10]}',
                    font=dict(size=9, color='#586069')
                ),
                tickfont=dict(size=8, color='#6a737d')
            ),
            plot_bgcolor='#fafbfc',
            paper_bgcolor='white',
            hovermode='x unified',
            uirevision='constant'
        )
        
        # Create card with plot
        card = html.Div([
            html.Div([
                html.H5(f'Channel {i} - {template_names[i]}', style={'margin': '0.25rem 0.5rem'}),
                html.Div([
                    dcc.Graph(
                        id={'type': 'channel-plot', 'index': i},
                        figure=fig,
                        config={
                            'displayModeBar': False,
                            'staticPlot': False,
                            'responsive': True
                        },
                        className='plot-bleed',
                        style={'height': f'{plot_height}px'}
                    )
                ])
            ], className='card-body', style={'padding': '0.25rem 0.5rem'})
        ], className='card-compact', style={'marginBottom': '0.5rem'})
        
        plots.append(card)
    
    return plots


# Callback for discord analysis
@app.callback(
    Output('discord-results-container', 'children'),
    Output('analysis-status', 'children'),
    Input('analyze-btn', 'n_clicks'),
    State({'type': 'channel-checkbox', 'index': dash.dependencies.ALL}, 'value'),
    State('analysis-m-input', 'value'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def run_discord_analysis(n_clicks, channel_values, m, data):
    if data is None:
        return None, "⚠️ Generate data first"
    
    # Get selected channels
    selected_channels = []
    for i, val in enumerate(channel_values):
        if val and len(val) > 0:
            selected_channels.append(i)
    
    if len(selected_channels) == 0:
        return None, "⚠️ Select at least one channel"
    
    # Get series data
    series = np.array(data['series'])
    N = data['N']
    
    # Generate discord profile
    try:
        status_msg = html.Div([
            html.Span(className='loading-spinner'),
            f"Analyzing channels {selected_channels}..."
        ])
        
        prof = discord_profile(selected_channels, N)
        
        # Calculate scores for each subset
        scores_by_arity = {}
        all_results = []
        
        for subset in prof:
            if len(subset) == 0:
                continue
            
            try:
                score = get_discord_score(series, subset=subset, m=m)
                arity = len(subset)
                
                if arity not in scores_by_arity:
                    scores_by_arity[arity] = []
                scores_by_arity[arity].append(score)
                
                all_results.append({
                    'subset': subset,
                    'arity': arity,
                    'score': score
                })
            except Exception as e:
                print(f"Error calculating score for subset {subset}: {e}")
                continue
        
        # Calculate mean scores by arity
        mean_scores = {}
        for arity, scores in scores_by_arity.items():
            mean_scores[arity] = np.mean(scores)
        
        # Create visualization
        arities = sorted(mean_scores.keys())
        means = [mean_scores[a] for a in arities]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=arities,
            y=means,
            marker=dict(
                color=means,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Mean Score')
            ),
            text=[f'{m:.3f}' for m in means],
            textposition='auto',
            hovertemplate='Arity: %{x}<br>Mean Score: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'Discord Profile Analysis - Channels {selected_channels}',
                font=dict(size=16, weight=600)
            ),
            xaxis=dict(
                title='Channel Set Arity',
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            yaxis=dict(
                title='Mean Anomaly Score'
            ),
            plot_bgcolor='#fafbfc',
            paper_bgcolor='white',
            height=400,
            margin=dict(l=60, r=40, t=60, b=60)
        )
        
        # Create detailed results table
        table_rows = []
        for result in sorted(all_results, key=lambda x: x['score'], reverse=True)[:20]:
            table_rows.append(
                html.Tr([
                    html.Td(str(sorted(result['subset'])), style={'padding': '0.5rem', 'borderBottom': '1px solid #e1e4e8'}),
                    html.Td(str(result['arity']), style={'padding': '0.5rem', 'borderBottom': '1px solid #e1e4e8', 'textAlign': 'center'}),
                    html.Td(f"{result['score']:.4f}", style={'padding': '0.5rem', 'borderBottom': '1px solid #e1e4e8', 'textAlign': 'right'})
                ])
            )
        
        results_div = html.Div([
            html.Div('Discord Profile Results', className='analysis-header'),
            
            html.Div([
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': False}
                )
            ], style={'marginBottom': '1.5rem'}),
            
            html.Div([
                html.H4('Top 20 Anomaly Scores', style={'fontSize': '14px', 'fontWeight': 600, 'marginBottom': '0.75rem'}),
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th('Channel Subset', style={'padding': '0.5rem', 'borderBottom': '2px solid #24292e', 'textAlign': 'left'}),
                            html.Th('Arity', style={'padding': '0.5rem', 'borderBottom': '2px solid #24292e', 'textAlign': 'center'}),
                            html.Th('Score', style={'padding': '0.5rem', 'borderBottom': '2px solid #24292e', 'textAlign': 'right'})
                        ])
                    ]),
                    html.Tbody(table_rows)
                ], style={'width': '100%', 'fontSize': '13px', 'borderCollapse': 'collapse'})
            ])
            
        ], className='discord-analysis-section')
        
        status = f"✓ Analysis complete - {len(prof)} subsets analyzed"
        
        return results_div, status
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# Fast hover callback using pattern matching
@app.callback(
    Output('hover-info', 'children'),
    Output('hover-info', 'className'),
    Output({'type': 'channel-plot', 'index': dash.dependencies.ALL}, 'figure'),
    Input({'type': 'channel-plot', 'index': dash.dependencies.ALL}, 'hoverData'),
    State({'type': 'channel-plot', 'index': dash.dependencies.ALL}, 'figure'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def update_hover(hover_data_list, figure_list, data):
    if data is None or not figure_list:
        return '', 'hover-info', figure_list
    
    series = np.array(data['series'])
    N = data['N']
    T = data['T']
    events_log = [(s, e, frozenset(c)) for s, e, c in data['events_log']]
    k_way_start = data['k_way_start']
    k_way_end = data['k_way_end']
    
    # Determine hover time
    hover_time = None
    hover_channels = None
    
    if hover_data_list:
        for hover_data in hover_data_list:
            if hover_data and 'points' in hover_data and len(hover_data['points']) > 0:
                hover_time = int(hover_data['points'][0]['x'])
                # Find which channels are active at this time
                for start, end, channels in events_log:
                    if start <= hover_time < end:
                        hover_channels = channels
                        break
                break
    
    # If no hover, return original figures
    if hover_time is None:
        # Remove any existing yellow highlights from all figures
        updated_figures = []
        for i, fig in enumerate(figure_list):
            new_fig = go.Figure(fig)
            # Keep only non-highlight shapes
            new_fig.layout.shapes = [
                shape for shape in (new_fig.layout.shapes or [])
                if not (hasattr(shape, 'name') and shape.name == 'highlight')
            ]
            # Re-add k-way region if it was removed
            has_k_way = any(
                shape.x0 == k_way_start and shape.x1 == k_way_end
                for shape in (new_fig.layout.shapes or [])
            )
            if not has_k_way:
                new_fig.add_shape(
                    type="rect",
                    x0=k_way_start,
                    x1=k_way_end,
                    y0=series[i].min() - 0.1,
                    y1=series[i].max() + 0.1,
                    fillcolor="rgba(220, 53, 69, 0.12)",
                    line=dict(width=0),
                    layer="below",
                    name="k_way_region"
                )
            updated_figures.append(new_fig)
        return '', 'hover-info', updated_figures
    
    # Find all matching events
    matching_events = []
    if hover_channels:
        matching_events = [(start, end) for start, end, channels in events_log 
                          if channels == hover_channels]
    
    # Update figures with highlights
    updated_figures = []
    for i, fig in enumerate(figure_list):
        new_fig = go.Figure(fig)
        
        # Remove old highlight shapes but keep k-way region
        new_fig.layout.shapes = [
            shape for shape in (new_fig.layout.shapes or [])
            if not (hasattr(shape, 'name') and shape.name == 'highlight')
        ]
        
        # Add yellow highlights for matching events
        for start, end in matching_events:
            new_fig.add_shape(
                type="rect",
                x0=start,
                x1=end,
                y0=series[i].min() - 0.1,
                y1=series[i].max() + 0.1,
                fillcolor="rgba(255, 243, 205, 0.6)",
                line=dict(color="rgba(255, 193, 7, 0.8)", width=2),
                layer="above",
                name="highlight"
            )
        
        updated_figures.append(new_fig)
    
    # Update hover info text
    hover_text = ''
    hover_class = 'hover-info'
    
    if hover_time is not None and hover_channels:
        channels_str = ', '.join(map(str, sorted(hover_channels)))
        hover_text = f'Time: {hover_time} • Channels: {{{channels_str}}} • Matches: {len(matching_events)}'
        hover_class = 'hover-info active'
    elif hover_time is not None:
        hover_text = f'Time: {hover_time} • No event'
        hover_class = 'hover-info active'
    
    return hover_text, hover_class, updated_figures

#Callback for downloading dataset
@app.callback(
    Output('download-npy', 'data'),
    Input('download-npy-btn', 'n_clicks'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def download_npy(n_clicks, data):
    if not n_clicks or data is None:
        return dash.no_update

    series = np.array(data['series'])  # shape: (N, T)

    buffer = io.BytesIO()
    np.save(buffer, series)
    buffer.seek(0)

    encoded = base64.b64encode(buffer.read()).decode('utf-8')

    return dict(
        content=encoded,
        filename='series.npy',
        type='application/octet-stream',
        base64=True
    )
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
