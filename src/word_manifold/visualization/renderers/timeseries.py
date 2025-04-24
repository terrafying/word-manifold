"""
Time Series Visualization Renderer.

This module provides renderers for time series visualizations,
supporting both local matplotlib-based rendering and server-side rendering.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

from ..base import VisualizationRenderer

class TimeSeriesRenderer(VisualizationRenderer):
    """Renderer for time series visualizations."""
    
    def __init__(self):
        """Initialize the renderer."""
        super().__init__()
        
    def render_local(
        self,
        data: Dict[str, Any],
        output_path: Path,
        title: Optional[str] = None,
        figure_size: tuple = (12, 6)
    ) -> Path:
        """Render time series visualization using matplotlib."""
        # Create figure with subplots
        fig = plt.figure(figsize=figure_size)
        
        # Main time series plot
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        
        time_points = data['time_points']
        patterns = data['patterns']
        metadata = data.get('metadata', {})
        
        # Plot each term's pattern with enhanced styling
        for term, pattern_data in patterns.items():
            values = pattern_data['values']
            meta = pattern_data['metadata']
            
            # Plot main line
            line = ax1.plot(time_points, values, label=term, alpha=0.7)
            color = line[0].get_color()
            
            # Add confidence interval if available
            if 'confidence' in meta:
                confidence = meta['confidence']
                ax1.fill_between(
                    time_points,
                    values - confidence * meta['std'],
                    values + confidence * meta['std'],
                    color=color, alpha=0.2
                )
        
        # Set title and labels
        ax1.set_title(title or f"Temporal Evolution of Terms ({metadata.get('timeframe', '')})")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Semantic Intensity")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # Correlation heatmap (if multiple terms)
        if len(patterns) > 1 and data.get('correlations'):
            ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
            terms = list(patterns.keys())
            corr_matrix = np.zeros((len(terms), len(terms)))
            
            for i, term1 in enumerate(terms):
                for j, term2 in enumerate(terms):
                    if i < j:
                        corr = data['correlations'].get(f"{term1}-{term2}", 0)
                        corr_matrix[i,j] = corr_matrix[j,i] = corr
                    elif i == j:
                        corr_matrix[i,j] = 1.0
                        
            sns.heatmap(
                corr_matrix,
                ax=ax2,
                xticklabels=terms,
                yticklabels=terms,
                cmap='RdYlBu_r',
                center=0,
                annot=True,
                fmt='.2f'
            )
            ax2.set_title("Term Correlations")
            plt.setp(ax2.get_xticklabels(), rotation=45)
            plt.setp(ax2.get_yticklabels(), rotation=0)
        
        # Pattern characteristics summary
        ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        ax3.axis('off')
        
        summary_text = []
        for term, pattern_data in patterns.items():
            meta = pattern_data['metadata']
            stats = [
                f"Term: {term}",
                f"Trend: {meta['trend']}",
                f"Volatility: {meta['volatility']:.2f}"
            ]
            
            # Add pattern-specific metrics
            if 'phase' in meta:
                stats.append(f"Phase: {meta['phase']:.2f}")
            elif 'slope' in meta:
                stats.append(f"Slope: {meta['slope']:.2f}")
            elif 'harmonics' in meta:
                stats.append(f"Harmonics: {meta['harmonics']}")
            elif 'frequency' in meta:
                stats.append(f"Frequency: {meta['frequency']:.2f}")
            elif 'components' in meta:
                stats.append(f"Components: {meta['components']}")
                
            summary_text.append(" | ".join(stats))
            
        ax3.text(
            0.5, 0.5,
            "\n".join(summary_text),
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )
        
        plt.tight_layout()
        
        # Save visualization
        output_file = output_path / f"timeseries_{metadata.get('timeframe', 'default')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
        
    def render_server(
        self,
        data: Dict[str, Any],
        server_url: str,
        endpoint: str = '/api/v1/visualize/timeseries'
    ) -> Dict[str, Any]:
        """Render time series visualization using server."""
        from ..utils import handle_server_request
        
        response = handle_server_request('POST', endpoint, {
            'data': data,
            'context': {
                'temporal_scale': data['metadata']['temporal']['scale'],
                'granularity': data['metadata']['interval'],
                'cyclic_nature': data['metadata']['pattern_type'] == 'cyclic'
            }
        })
        
        return response
        
    def render_insights(
        self,
        data: Dict[str, Any],
        hexagram_data: Optional[Dict[str, Any]] = None,
        output_path: Path = None
    ) -> Optional[str]:
        """Render temporal insights, optionally including I Ching analysis."""
        insights = []
        
        # Temporal context
        temporal = data['metadata']['temporal']
        insights.append("Temporal Context")
        insights.append("================")
        insights.append(f"Scale: {temporal['scale']}")
        insights.append(f"Period: {temporal['start_time']} to {temporal['end_time']}")
        insights.append(f"Interval: {temporal['interval']}")
        insights.append(f"Number of points: {temporal['num_points']}")
        insights.append("")
        
        # Pattern analysis
        insights.append("Pattern Analysis")
        insights.append("================")
        for term, pattern_data in data['patterns'].items():
            meta = pattern_data['metadata']
            insights.append(f"\n{term}:")
            insights.append(f"  Trend: {meta['trend']}")
            insights.append(f"  Mean: {meta['mean']:.3f}")
            insights.append(f"  Standard Deviation: {meta['std']:.3f}")
            insights.append(f"  Volatility: {meta['volatility']:.3f}")
            
            # Pattern-specific insights
            if 'phase' in meta:
                insights.append(f"  Phase Shift: {meta['phase']:.2f} radians")
            elif 'slope' in meta:
                insights.append(f"  Slope: {meta['slope']:.2f}")
                insights.append(f"  Confidence: {meta['confidence']:.2f}")
            elif 'harmonics' in meta:
                insights.append(f"  Number of Harmonics: {meta['harmonics']}")
            elif 'frequency' in meta:
                insights.append(f"  Base Frequency: {meta['frequency']:.2f}")
            elif 'components' in meta:
                insights.append(f"  Wave Components: {meta['components']}")
        
        # Correlation analysis
        if data.get('correlations'):
            insights.append("\nTerm Correlations")
            insights.append("================")
            for pair, corr in data['correlations'].items():
                term1, term2 = pair.split('-')
                relationship = (
                    "strong positive" if corr > 0.7 else
                    "moderate positive" if corr > 0.3 else
                    "weak positive" if corr > 0 else
                    "strong negative" if corr < -0.7 else
                    "moderate negative" if corr < -0.3 else
                    "weak negative"
                )
                insights.append(f"{term1} ↔ {term2}: {relationship} ({corr:.2f})")
        
        # I Ching insights
        if hexagram_data:
            insights.append("\nI Ching Analysis")
            insights.append("===============")
            hexagram = hexagram_data['hexagram']
            changing_lines = hexagram_data['changing_lines']
            trigram_mapping = hexagram_data['trigram_mapping']
            
            insights.append(f"\nHexagram {hexagram.number}: {hexagram.name}")
            
            lower_trigram, upper_trigram = hexagram.get_trigrams()
            insights.append("\nTemporal Foundation (Lower Trigram):")
            insights.append(f"  {', '.join(trigram_mapping.get(str(lower_trigram), []))}")
            
            insights.append("\nTemporal Direction (Upper Trigram):")
            insights.append(f"  {', '.join(trigram_mapping.get(str(upper_trigram), []))}")
            
            if changing_lines:
                insights.append("\nTransformation Points:")
                insights.append(f"  Lines {', '.join(map(str, changing_lines))} indicate temporal shifts")
        
        insight_text = "\n".join(insights)
        
        if output_path:
            insight_file = output_path / "temporal_insights.txt"
            insight_file.write_text(insight_text)
            return str(insight_file)
            
        return insight_text 