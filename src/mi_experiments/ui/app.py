import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from typing import Optional

from src.analysis.viz import VisualizationTools
from src.analysis.attention import AttentionAnalyzer
from src.analysis.tokens import TokenAnalyzer
from src.utils.emphasis import EmphasisConfig, EmphasisManager

class DashApp:
    def __init__(
        self,
        model,
        tokenizer,
        attention_analyzer: Optional[AttentionAnalyzer] = None,
        token_analyzer: Optional[TokenAnalyzer] = None
    ):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.model = model
        self.tokenizer = tokenizer
        self.attention_analyzer = attention_analyzer or AttentionAnalyzer(model, tokenizer)
        self.token_analyzer = token_analyzer or TokenAnalyzer(model, tokenizer)
        self.emphasis_manager = EmphasisManager(model)
        
        self.app.layout = self._create_layout()
        self._init_callbacks()
        
    def _create_layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Model Analysis Dashboard"),
                    html.Hr()
                ])
            ]),
            
            # Input Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Input Text"),
                        dbc.CardBody([
                            dbc.Textarea(
                                id="input-text",
                                placeholder="Enter text to analyze...",
                                style={"height": "150px"}
                            ),
                            dbc.Button(
                                "Analyze",
                                id="analyze-button",
                                color="primary",
                                className="mt-3"
                            )
                        ])
                    ])
                ], width=6),
                
                # Configuration Section
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Analysis Configuration"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Layer:"),
                                    dcc.Dropdown(
                                        id="layer-select",
                                        options=[
                                            {"label": f"Layer {i}", "value": i}
                                            for i in range(len(self.model.layers))
                                        ]
                                    )
                                ]),
                                dbc.Col([
                                    html.Label("Head:"),
                                    dcc.Dropdown(
                                        id="head-select",
                                        options=[
                                            {"label": f"Head {i}", "value": i}
                                            for i in range(self.model.layers[0].self_attn.n_heads)
                                        ]
                                    )
                                ])
                            ])
                        ])
                    ])
                ], width=6)
            ]),
            
            # Visualization Section
            dbc.Row([
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(id="attention-plot")
                        ], label="Attention Patterns"),
                        
                        dbc.Tab([
                            dcc.Graph(id="embeddings-plot")
                        ], label="Token Embeddings"),
                        
                        dbc.Tab([
                            dcc.Graph(id="activations-plot")
                        ], label="Layer Activations")
                    ])
                ])
            ], className="mt-4")
        ], fluid=True)
    
    def _init_callbacks(self):
        @self.app.callback(
            [Output("attention-plot", "figure"),
             Output("embeddings-plot", "figure"),
             Output("activations-plot", "figure")],
            [Input("analyze-button", "n_clicks")],
            [State("input-text", "value"),
             State("layer-select", "value"),
             State("head-select", "value")]
        )
        def update_plots(n_clicks, text, layer_idx, head_idx):
            if not text:
                return dash.no_update
                
            # Get analysis results
            attention_patterns = self.attention_analyzer.analyze_attention(
                text, layer_idx, head_idx
            )
            token_analysis = self.token_analyzer.analyze_sequence(text)
            
            # Create visualizations
            attention_fig = VisualizationTools.plot_attention_patterns(
                attention_patterns.patterns,
                attention_patterns.tokens,
                layer_idx,
                head_idx
            )
            
            embeddings_fig = VisualizationTools.plot_token_embeddings(
                token_analysis.embeddings,
                token_analysis.token_text
            )
            
            activations_fig = VisualizationTools.plot_layer_activations(
                token_analysis.layer_states[layer_idx] if layer_idx is not None else token_analysis.layer_states[-1],
                layer_idx if layer_idx is not None else len(self.model.layers) - 1,
                token_analysis.token_text
            )
            
            return attention_fig, embeddings_fig, activations_fig
    
    def run_server(self, **kwargs):
        self.app.run_server(**kwargs)
