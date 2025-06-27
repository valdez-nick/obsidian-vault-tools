"""
Dashboard Generator Implementation

Automated dashboard creation with interactive visualizations, real-time metrics,
and customizable layouts for PM performance monitoring.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import base64
from io import BytesIO

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Types of charts available."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    CANDLESTICK = "candlestick"
    FUNNEL = "funnel"
    SANKEY = "sankey"
    TREEMAP = "treemap"
    RADAR = "radar"
    BOX = "box"


class DashboardLayout(Enum):
    """Dashboard layout options."""
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    GRID = "grid"
    TABS = "tabs"
    SIDEBAR = "sidebar"


class RefreshRate(Enum):
    """Dashboard refresh rates."""
    REAL_TIME = 1  # seconds
    NEAR_REAL_TIME = 10
    MINUTE = 60
    FIVE_MINUTES = 300
    HOURLY = 3600
    DAILY = 86400


@dataclass
class MetricCard:
    """Configuration for a metric card."""
    title: str
    value: Union[float, int, str]
    previous_value: Optional[Union[float, int]] = None
    unit: Optional[str] = None
    change_type: Optional[str] = None  # 'increase', 'decrease', 'neutral'
    icon: Optional[str] = None
    color: Optional[str] = None
    sparkline_data: Optional[List[float]] = None
    
    def calculate_change(self) -> Tuple[float, str]:
        """Calculate change percentage and direction."""
        if self.previous_value is not None and isinstance(self.value, (int, float)):
            change = ((self.value - self.previous_value) / self.previous_value) * 100
            direction = "increase" if change > 0 else "decrease" if change < 0 else "neutral"
            return change, direction
        return 0.0, "neutral"


@dataclass
class MetricVisualization:
    """Configuration for a metric visualization."""
    name: str
    chart_type: ChartType
    data: pd.DataFrame
    x_column: Optional[str] = None
    y_columns: List[str] = field(default_factory=list)
    title: Optional[str] = None
    subtitle: Optional[str] = None
    color_scheme: Optional[str] = None
    height: int = 400
    width: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'chart_type': self.chart_type.value,
            'title': self.title,
            'subtitle': self.subtitle,
            'height': self.height,
            'width': self.width,
            'config': self.config
        }


@dataclass
class DashboardSection:
    """A section of the dashboard."""
    name: str
    title: str
    metric_cards: List[MetricCard] = field(default_factory=list)
    visualizations: List[MetricVisualization] = field(default_factory=list)
    layout: DashboardLayout = DashboardLayout.SINGLE_COLUMN
    description: Optional[str] = None


class DashboardGenerator:
    """
    Generates interactive dashboards for PM performance monitoring.
    
    Supports multiple visualization libraries and export formats.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Dashboard Generator.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        self.theme = config.get('theme', 'light')
        self.output_path = Path(config.get('output_path', './dashboards'))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Visualization settings
        self.default_colors = config.get('colors', [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ])
        
        # Configure plotting libraries
        if PLOTLY_AVAILABLE:
            pio.templates.default = "plotly_white" if self.theme == 'light' else "plotly_dark"
        
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8-whitegrid' if self.theme == 'light' else 'dark_background')
            
    def create_metric_card_html(self, card: MetricCard) -> str:
        """
        Create HTML for a metric card.
        
        Args:
            card: Metric card configuration
            
        Returns:
            HTML string
        """
        change, direction = card.calculate_change()
        
        # Determine color based on change
        if card.color:
            color = card.color
        elif direction == "increase":
            color = "#2ca02c" if card.change_type != "negative" else "#d62728"
        elif direction == "decrease":
            color = "#d62728" if card.change_type != "negative" else "#2ca02c"
        else:
            color = "#7f7f7f"
        
        # Create sparkline if data provided
        sparkline_html = ""
        if card.sparkline_data and PLOTLY_AVAILABLE:
            fig = go.Figure(go.Scatter(
                y=card.sparkline_data,
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            ))
            fig.update_layout(
                height=50,
                width=150,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            sparkline_html = fig.to_html(div_id=f"sparkline_{card.title.replace(' ', '_')}", include_plotlyjs=False)
        
        html = f"""
        <div class="metric-card" style="
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px;
        ">
            <h3 style="margin: 0 0 10px 0; color: #333; font-size: 16px;">{card.title}</h3>
            <div style="display: flex; align-items: baseline;">
                <span style="font-size: 32px; font-weight: bold; color: {color};">
                    {card.value}{card.unit or ''}
                </span>
                {f'<span style="margin-left: 10px; font-size: 14px; color: {color};">({change:+.1f}%)</span>' if card.previous_value is not None else ''}
            </div>
            {sparkline_html}
        </div>
        """
        
        return html
    
    def create_visualization(self, viz: MetricVisualization) -> Union[Any, None]:
        """
        Create a visualization based on configuration.
        
        Args:
            viz: Visualization configuration
            
        Returns:
            Plotly figure, matplotlib figure, or None
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for visualization")
            return None
        
        if viz.chart_type == ChartType.LINE:
            return self._create_line_chart(viz)
        elif viz.chart_type == ChartType.BAR:
            return self._create_bar_chart(viz)
        elif viz.chart_type == ChartType.SCATTER:
            return self._create_scatter_plot(viz)
        elif viz.chart_type == ChartType.PIE:
            return self._create_pie_chart(viz)
        elif viz.chart_type == ChartType.HEATMAP:
            return self._create_heatmap(viz)
        elif viz.chart_type == ChartType.GAUGE:
            return self._create_gauge_chart(viz)
        elif viz.chart_type == ChartType.RADAR:
            return self._create_radar_chart(viz)
        elif viz.chart_type == ChartType.BOX:
            return self._create_box_plot(viz)
        else:
            logger.warning(f"Unsupported chart type: {viz.chart_type}")
            return None
    
    def _create_line_chart(self, viz: MetricVisualization):
        """Create line chart."""
        fig = go.Figure()
        
        for y_col in viz.y_columns:
            fig.add_trace(go.Scatter(
                x=viz.data[viz.x_column] if viz.x_column else viz.data.index,
                y=viz.data[y_col],
                mode='lines+markers',
                name=y_col,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=viz.title or "Line Chart",
            xaxis_title=viz.x_column or "Index",
            yaxis_title="Value",
            height=viz.height,
            width=viz.width,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_bar_chart(self, viz: MetricVisualization):
        """Create bar chart."""
        fig = go.Figure()
        
        for i, y_col in enumerate(viz.y_columns):
            fig.add_trace(go.Bar(
                x=viz.data[viz.x_column] if viz.x_column else viz.data.index,
                y=viz.data[y_col],
                name=y_col,
                marker_color=self.default_colors[i % len(self.default_colors)]
            ))
        
        fig.update_layout(
            title=viz.title or "Bar Chart",
            xaxis_title=viz.x_column or "Category",
            yaxis_title="Value",
            height=viz.height,
            width=viz.width,
            barmode=viz.config.get('barmode', 'group')
        )
        
        return fig
    
    def _create_scatter_plot(self, viz: MetricVisualization):
        """Create scatter plot."""
        fig = go.Figure()
        
        if len(viz.y_columns) >= 2:
            # Use first two columns for x and y
            x_col, y_col = viz.y_columns[0], viz.y_columns[1]
            size_col = viz.y_columns[2] if len(viz.y_columns) > 2 else None
            
            fig.add_trace(go.Scatter(
                x=viz.data[x_col],
                y=viz.data[y_col],
                mode='markers',
                marker=dict(
                    size=viz.data[size_col] * 10 if size_col else 10,
                    color=viz.data[size_col] if size_col else self.default_colors[0],
                    colorscale='Viridis',
                    showscale=True if size_col else False
                ),
                text=viz.data.index,
                name="Data Points"
            ))
            
            fig.update_layout(
                title=viz.title or "Scatter Plot",
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=viz.height,
                width=viz.width
            )
        
        return fig
    
    def _create_pie_chart(self, viz: MetricVisualization):
        """Create pie chart."""
        if viz.y_columns:
            values_col = viz.y_columns[0]
            labels_col = viz.x_column or viz.data.index
            
            fig = go.Figure(data=[go.Pie(
                labels=viz.data[labels_col] if isinstance(labels_col, str) else labels_col,
                values=viz.data[values_col],
                hole=viz.config.get('hole', 0)  # 0 for pie, >0 for donut
            )])
            
            fig.update_layout(
                title=viz.title or "Pie Chart",
                height=viz.height,
                width=viz.width
            )
            
            return fig
        
        return go.Figure()
    
    def _create_heatmap(self, viz: MetricVisualization):
        """Create heatmap."""
        # Use all numeric columns if not specified
        if not viz.y_columns:
            numeric_cols = viz.data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = viz.y_columns
        
        # Calculate correlation matrix
        corr_matrix = viz.data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title=viz.title or "Correlation Heatmap",
            height=viz.height,
            width=viz.width or viz.height  # Square by default
        )
        
        return fig
    
    def _create_gauge_chart(self, viz: MetricVisualization):
        """Create gauge chart."""
        if viz.y_columns:
            value = viz.data[viz.y_columns[0]].iloc[-1]  # Latest value
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': viz.title or viz.y_columns[0]},
                delta={'reference': viz.config.get('target', value * 0.9)},
                gauge={
                    'axis': {'range': [None, viz.config.get('max_value', value * 1.5)]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, value * 0.5], 'color': "lightgray"},
                        {'range': [value * 0.5, value * 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': viz.config.get('threshold', value * 1.2)
                    }
                }
            ))
            
            fig.update_layout(height=viz.height, width=viz.width)
            return fig
        
        return go.Figure()
    
    def _create_radar_chart(self, viz: MetricVisualization):
        """Create radar chart."""
        fig = go.Figure()
        
        for i, col in enumerate(viz.y_columns):
            fig.add_trace(go.Scatterpolar(
                r=viz.data[col],
                theta=viz.data[viz.x_column] if viz.x_column else viz.data.index,
                fill='toself',
                name=col,
                line_color=self.default_colors[i % len(self.default_colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, viz.data[viz.y_columns].max().max()]
                )
            ),
            showlegend=True,
            title=viz.title or "Radar Chart",
            height=viz.height,
            width=viz.width
        )
        
        return fig
    
    def _create_box_plot(self, viz: MetricVisualization):
        """Create box plot."""
        fig = go.Figure()
        
        for col in viz.y_columns:
            fig.add_trace(go.Box(
                y=viz.data[col],
                name=col,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title=viz.title or "Box Plot",
            yaxis_title="Value",
            height=viz.height,
            width=viz.width
        )
        
        return fig
    
    def generate_dashboard(self, sections: List[DashboardSection], title: str = "PM Analytics Dashboard") -> str:
        """
        Generate complete dashboard HTML.
        
        Args:
            sections: List of dashboard sections
            title: Dashboard title
            
        Returns:
            Path to generated dashboard HTML file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_file = self.output_path / f"dashboard_{timestamp}.html"
        
        # Build HTML structure
        html_parts = [self._generate_html_header(title)]
        
        # Add navigation if multiple sections
        if len(sections) > 1:
            html_parts.append(self._generate_navigation(sections))
        
        # Add each section
        for section in sections:
            html_parts.append(self._generate_section_html(section))
        
        # Add footer
        html_parts.append(self._generate_html_footer())
        
        # Write to file
        with open(dashboard_file, 'w') as f:
            f.write('\n'.join(html_parts))
        
        logger.info(f"Dashboard generated: {dashboard_file}")
        return str(dashboard_file)
    
    def _generate_html_header(self, title: str) -> str:
        """Generate HTML header with styles and scripts."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: {'#f5f5f5' if self.theme == 'light' else '#1a1a1a'};
                    color: {'#333' if self.theme == 'light' else '#f5f5f5'};
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: {'white' if self.theme == 'light' else '#2a2a2a'};
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .section {{
                    background: {'white' if self.theme == 'light' else '#2a2a2a'};
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .metric-card {{
                    background: {'#f8f9fa' if self.theme == 'light' else '#3a3a3a'};
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .chart-container {{
                    margin: 20px 0;
                }}
                .navigation {{
                    display: flex;
                    gap: 10px;
                    margin-bottom: 20px;
                    flex-wrap: wrap;
                }}
                .nav-button {{
                    padding: 10px 20px;
                    background: {'#007bff' if self.theme == 'light' else '#0056b3'};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    text-decoration: none;
                }}
                .nav-button:hover {{
                    background: {'#0056b3' if self.theme == 'light' else '#007bff'};
                }}
                .timestamp {{
                    text-align: right;
                    color: {'#666' if self.theme == 'light' else '#999'};
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{title}</h1>
                    <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
        """
    
    def _generate_navigation(self, sections: List[DashboardSection]) -> str:
        """Generate navigation for sections."""
        nav_html = '<div class="navigation">'
        
        for section in sections:
            section_id = section.name.replace(' ', '_').lower()
            nav_html += f'<a href="#{section_id}" class="nav-button">{section.title}</a>'
        
        nav_html += '</div>'
        return nav_html
    
    def _generate_section_html(self, section: DashboardSection) -> str:
        """Generate HTML for a dashboard section."""
        section_id = section.name.replace(' ', '_').lower()
        html = f'<div id="{section_id}" class="section">'
        html += f'<h2>{section.title}</h2>'
        
        if section.description:
            html += f'<p>{section.description}</p>'
        
        # Add metric cards
        if section.metric_cards:
            html += '<div class="metrics-grid">'
            for card in section.metric_cards:
                html += self.create_metric_card_html(card)
            html += '</div>'
        
        # Add visualizations
        for viz in section.visualizations:
            fig = self.create_visualization(viz)
            if fig:
                chart_id = f"chart_{section_id}_{viz.name.replace(' ', '_')}"
                html += f'<div id="{chart_id}" class="chart-container"></div>'
                
                # Add Plotly script to render the chart
                html += f'<script>Plotly.newPlot("{chart_id}", {fig.to_json()});</script>'
        
        html += '</div>'
        return html
    
    def _generate_html_footer(self) -> str:
        """Generate HTML footer."""
        return """
            </div>
            <script>
                // Auto-refresh functionality
                function autoRefresh(seconds) {
                    setTimeout(function() {
                        window.location.reload();
                    }, seconds * 1000);
                }
                
                // Responsive chart sizing
                window.addEventListener('resize', function() {
                    var charts = document.querySelectorAll('[id^="chart_"]');
                    charts.forEach(function(chart) {
                        Plotly.Plots.resize(chart);
                    });
                });
            </script>
        </body>
        </html>
        """
    
    def create_executive_dashboard(self, data: Dict[str, pd.DataFrame]) -> str:
        """
        Create an executive-level dashboard with key PM metrics.
        
        Args:
            data: Dictionary of dataframes with various metrics
            
        Returns:
            Path to generated dashboard
        """
        sections = []
        
        # Overview section
        if 'summary' in data:
            summary_df = data['summary']
            overview_cards = [
                MetricCard(
                    title="Team Velocity",
                    value=summary_df['velocity'].iloc[-1],
                    previous_value=summary_df['velocity'].iloc[-2] if len(summary_df) > 1 else None,
                    unit=" pts/sprint",
                    sparkline_data=summary_df['velocity'].tail(10).tolist()
                ),
                MetricCard(
                    title="On-Time Delivery",
                    value=summary_df['on_time_rate'].iloc[-1] * 100,
                    previous_value=summary_df['on_time_rate'].iloc[-2] * 100 if len(summary_df) > 1 else None,
                    unit="%",
                    change_type="positive"
                ),
                MetricCard(
                    title="Quality Score",
                    value=summary_df['quality_score'].iloc[-1],
                    previous_value=summary_df['quality_score'].iloc[-2] if len(summary_df) > 1 else None,
                    unit="/100",
                    change_type="positive"
                ),
                MetricCard(
                    title="Team Health",
                    value=summary_df['team_health'].iloc[-1],
                    previous_value=summary_df['team_health'].iloc[-2] if len(summary_df) > 1 else None,
                    unit="/10",
                    sparkline_data=summary_df['team_health'].tail(10).tolist()
                )
            ]
            
            overview_section = DashboardSection(
                name="overview",
                title="Executive Overview",
                metric_cards=overview_cards,
                description="Key performance indicators for PM teams"
            )
            sections.append(overview_section)
        
        # Performance trends section
        if 'performance' in data:
            perf_df = data['performance']
            
            trend_viz = MetricVisualization(
                name="velocity_trend",
                chart_type=ChartType.LINE,
                data=perf_df,
                x_column='date',
                y_columns=['velocity', 'velocity_target'],
                title="Velocity Trend vs Target",
                height=400
            )
            
            burndown_viz = MetricVisualization(
                name="burndown",
                chart_type=ChartType.LINE,
                data=perf_df,
                x_column='date',
                y_columns=['remaining_work', 'ideal_burndown'],
                title="Sprint Burndown",
                config={'line_dash_sequence': ['solid', 'dash']}
            )
            
            performance_section = DashboardSection(
                name="performance",
                title="Performance Trends",
                visualizations=[trend_viz, burndown_viz],
                layout=DashboardLayout.TWO_COLUMN
            )
            sections.append(performance_section)
        
        # Team analytics section
        if 'team' in data:
            team_df = data['team']
            
            workload_viz = MetricVisualization(
                name="workload_distribution",
                chart_type=ChartType.BAR,
                data=team_df,
                x_column='team_member',
                y_columns=['assigned_points', 'completed_points'],
                title="Team Workload Distribution",
                config={'barmode': 'group'}
            )
            
            # Check if we have week column, otherwise use team data directly
            if 'week' in team_df.columns:
                heatmap_data = team_df.pivot_table(values='productivity', index='week', columns='team_member')
            else:
                # Use correlation matrix of team metrics as alternative
                heatmap_data = team_df[['assigned_points', 'completed_points', 'productivity']]
            
            productivity_viz = MetricVisualization(
                name="productivity_heatmap",
                chart_type=ChartType.HEATMAP,
                data=heatmap_data,
                title="Team Productivity Analysis"
            )
            
            team_section = DashboardSection(
                name="team",
                title="Team Analytics",
                visualizations=[workload_viz, productivity_viz]
            )
            sections.append(team_section)
        
        # Risk indicators section
        if 'risks' in data:
            risk_df = data['risks']
            
            risk_gauge = MetricVisualization(
                name="burnout_risk",
                chart_type=ChartType.GAUGE,
                data=risk_df,
                y_columns=['burnout_risk_score'],
                title="Team Burnout Risk",
                config={'max_value': 100, 'threshold': 70}
            )
            
            risk_factors = MetricVisualization(
                name="risk_factors",
                chart_type=ChartType.RADAR,
                data=risk_df,
                x_column='risk_factor',
                y_columns=['current_score', 'threshold'],
                title="Risk Factor Analysis"
            )
            
            risk_section = DashboardSection(
                name="risks",
                title="Risk Indicators",
                visualizations=[risk_gauge, risk_factors],
                layout=DashboardLayout.TWO_COLUMN
            )
            sections.append(risk_section)
        
        return self.generate_dashboard(sections, "PM Executive Dashboard")


class AlertEngine:
    """
    Real-time alert engine for monitoring PM metrics and triggering notifications.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Alert Engine.
        
        Args:
            config: Alert configuration
        """
        self.config = config
        self.alert_rules = []
        self.alert_history = []
        self.notification_channels = config.get('notification_channels', [])
        
    def add_alert_rule(self, metric: str, condition: str, threshold: float, 
                      severity: str = "warning", message_template: str = None):
        """
        Add an alert rule.
        
        Args:
            metric: Metric to monitor
            condition: Condition ('>', '<', '==', '!=', 'contains')
            threshold: Threshold value
            severity: Alert severity ('info', 'warning', 'critical')
            message_template: Custom message template
        """
        rule = {
            'id': f"rule_{len(self.alert_rules) + 1}",
            'metric': metric,
            'condition': condition,
            'threshold': threshold,
            'severity': severity,
            'message_template': message_template or f"{metric} {condition} {threshold}",
            'enabled': True,
            'created_at': datetime.now()
        }
        
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule['id']}")
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check current metrics against alert rules.
        
        Args:
            metrics: Current metric values
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for rule in self.alert_rules:
            if not rule['enabled']:
                continue
            
            metric_value = metrics.get(rule['metric'])
            if metric_value is None:
                continue
            
            # Evaluate condition
            triggered = False
            if rule['condition'] == '>':
                triggered = metric_value > rule['threshold']
            elif rule['condition'] == '<':
                triggered = metric_value < rule['threshold']
            elif rule['condition'] == '==':
                triggered = metric_value == rule['threshold']
            elif rule['condition'] == '!=':
                triggered = metric_value != rule['threshold']
            elif rule['condition'] == 'contains' and isinstance(metric_value, str):
                triggered = str(rule['threshold']) in metric_value
            
            if triggered:
                alert = {
                    'rule_id': rule['id'],
                    'metric': rule['metric'],
                    'value': metric_value,
                    'threshold': rule['threshold'],
                    'severity': rule['severity'],
                    'message': rule['message_template'].format(
                        metric=rule['metric'],
                        value=metric_value,
                        threshold=rule['threshold']
                    ),
                    'timestamp': datetime.now()
                }
                
                triggered_alerts.append(alert)
                self.alert_history.append(alert)
                
                logger.warning(f"Alert triggered: {alert['message']}")
        
        return triggered_alerts
    
    def send_notifications(self, alerts: List[Dict[str, Any]]):
        """
        Send notifications for triggered alerts.
        
        Args:
            alerts: List of triggered alerts
        """
        for alert in alerts:
            for channel in self.notification_channels:
                try:
                    if channel['type'] == 'email':
                        self._send_email_notification(alert, channel)
                    elif channel['type'] == 'slack':
                        self._send_slack_notification(alert, channel)
                    elif channel['type'] == 'webhook':
                        self._send_webhook_notification(alert, channel)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel['type']}: {e}")
    
    def _send_email_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]):
        """Send email notification (placeholder)."""
        logger.info(f"Would send email to {channel['recipients']}: {alert['message']}")
    
    def _send_slack_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]):
        """Send Slack notification (placeholder)."""
        logger.info(f"Would send Slack message to {channel['webhook_url']}: {alert['message']}")
    
    def _send_webhook_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]):
        """Send webhook notification (placeholder)."""
        logger.info(f"Would send webhook to {channel['url']}: {alert}")
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of recent alerts.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Alert summary statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alert_history if a['timestamp'] > cutoff_time]
        
        summary = {
            'total_alerts': len(recent_alerts),
            'by_severity': {},
            'by_metric': {},
            'most_frequent': None
        }
        
        for alert in recent_alerts:
            # Count by severity
            severity = alert['severity']
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # Count by metric
            metric = alert['metric']
            summary['by_metric'][metric] = summary['by_metric'].get(metric, 0) + 1
        
        # Find most frequent alert
        if summary['by_metric']:
            summary['most_frequent'] = max(summary['by_metric'], key=summary['by_metric'].get)
        
        return summary