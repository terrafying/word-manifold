"""
Time Series Visualization Renderer.

This module provides renderers for time series visualizations,
supporting both local matplotlib-based rendering and cloud-based rendering.
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import logging
import sys
from importlib import import_module

# Dynamically import visualization modules
try:
    base = import_module('word_manifold.visualization.base')
    cloud = import_module('word_manifold.visualization.cloud_services')
    VisualizationRenderer = base.VisualizationRenderer
    CloudVisualizationService = cloud.CloudVisualizationService
except ImportError as e:
    # Fallback to relative imports if package is not installed
    from ..base import VisualizationRenderer
    from ..cloud_services import CloudVisualizationService

logger = logging.getLogger(__name__)

class TimeSeriesRenderer(VisualizationRenderer):
    """Renderer for time series visualizations."""
    
    def __init__(
        self,
        cloud_provider: Optional[str] = None,
        cloud_credentials: Optional[Dict[str, Any]] = None
    ):
        """Initialize the renderer.
        
        Args:
            cloud_provider: Optional cloud provider for remote visualization
            cloud_credentials: Credentials for cloud provider
        """
        super().__init__()
        # Initialize LLM components for semantic interpretation
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self._insights: List[str] = []
        
        # Initialize cloud service if provider specified
        self.cloud_service = None
        if cloud_provider:
            self.cloud_service = CloudVisualizationService(
                provider=cloud_provider,
                credentials=cloud_credentials
            )

    def _interpret_semantic_shift(
        self,
        initial_terms: List[str],
        initial_vectors: np.ndarray,
        final_vectors: np.ndarray,
        pattern_type: str
    ) -> Dict[str, Any]:
        """Interpret semantic shifts in the phase space using LLM."""
        # Calculate vector changes
        vector_changes = final_vectors - initial_vectors
        magnitudes = np.linalg.norm(vector_changes, axis=1)
        directions = vector_changes / magnitudes[:, np.newaxis]
        
        # Calculate pairwise similarities before and after
        initial_sim = cosine_similarity(initial_vectors)
        final_sim = cosine_similarity(final_vectors)
        sim_changes = final_sim - initial_sim
        
        # Prepare semantic interpretation prompt
        prompt = f"""Analyze the semantic evolution of concepts in a temporal phase space:

Initial Terms: {', '.join(initial_terms)}

Pattern Type: {pattern_type}

Vector Changes:
{' '.join([f'{term}: magnitude={mag:.3f}' for term, mag in zip(initial_terms, magnitudes)])}

Relationship Changes:
"""
        # Add significant relationship changes
        for i, term1 in enumerate(initial_terms):
            for j, term2 in enumerate(initial_terms[i+1:], i+1):
                if abs(sim_changes[i,j]) > 0.1:  # Only include significant changes
                    change = "increased" if sim_changes[i,j] > 0 else "decreased"
                    prompt += f"- Similarity between '{term1}' and '{term2}' has {change} by {abs(sim_changes[i,j]):.3f}\n"

        prompt += """
Based on these transformations:
1. How has the meaning of each concept evolved?
2. What new relationships or patterns have emerged?
3. What is the overall narrative of this semantic journey?
4. What metaphysical or philosophical insights can be drawn from these changes?

Please provide a detailed interpretation that connects mathematical patterns to semantic meaning."""

        # Generate interpretation using LLM
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        outputs = self.model.generate(
            **inputs,
            max_length=1000,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )
        
        interpretation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'vector_changes': {
                'magnitudes': magnitudes.tolist(),
                'directions': directions.tolist()
            },
            'similarity_changes': sim_changes.tolist(),
            'interpretation': interpretation
        }
        
    def _analyze_phase_space_topology(
        self,
        patterns: Dict[str, Dict[str, Any]],
        pattern_type: str
    ) -> Dict[str, Any]:
        """Analyze the topology and dynamics of the phase space."""
        terms = list(patterns.keys())
        values = np.array([patterns[term]['values'] for term in terms])
        
        # Get initial and final states
        initial_vectors = values[:, 0]
        final_vectors = values[:, -1]
        
        # Calculate phase space characteristics
        characteristics = {
            'dimensionality': len(terms),
            'volume': np.prod([np.ptp(values[i]) for i in range(len(terms))]),
            'centroid': np.mean(values, axis=1).tolist(),
            'trajectory_lengths': [np.sum(np.sqrt(np.sum(np.diff(values[i])**2))) for i in range(len(terms))],
            'pattern_type': pattern_type
        }
        
        # Analyze dynamics
        if pattern_type == 'cyclic':
            # Check for periodic behavior
            for i, term_values in enumerate(values):
                fft = np.fft.fft(term_values)
                freqs = np.fft.fftfreq(len(term_values))
                main_freq_idx = np.argmax(np.abs(fft))
                characteristics[f'{terms[i]}_frequency'] = float(freqs[main_freq_idx])
                
        elif pattern_type == 'linear':
            # Check for convergence/divergence
            for i, term_values in enumerate(values):
                slope = np.polyfit(range(len(term_values)), term_values, 1)[0]
                characteristics[f'{terms[i]}_trend'] = float(slope)
                
        # Get semantic interpretation
        interpretation = self._interpret_semantic_shift(
            terms,
            initial_vectors,
            final_vectors,
            pattern_type
        )
        
        characteristics.update({
            'semantic_analysis': interpretation
        })
        
        return characteristics

    def render_local(
        self,
        data: Dict[str, Any],
        output_path: Path,
        title: Optional[str] = None,
        figure_size: tuple = (15, 10),
        interactive: bool = True,
        upload_to_cloud: bool = False
    ) -> Path:
        """Render time series visualization using matplotlib or plotly.
        
        Args:
            data: Visualization data
            output_path: Output directory
            title: Optional title
            figure_size: Figure dimensions
            interactive: Whether to create interactive visualization
            upload_to_cloud: Whether to upload result to cloud storage
            
        Returns:
            Path to visualization file
        """
        output_file = (
            self._render_interactive(data, output_path, title)
            if interactive else
            self._render_static(data, output_path, title, figure_size)
        )
        
        # Upload to cloud if requested and cloud service is configured
        if upload_to_cloud and self.cloud_service:
            try:
                # Upload with metadata
                result = self.cloud_service.upload_visualization(
                    output_file,
                    metadata={
                        'type': 'timeseries',
                        'interactive': interactive,
                        'temporal_scale': data['metadata']['temporal']['scale'],
                        'pattern_type': data['metadata']['pattern_type'],
                        'terms': list(data['patterns'].keys())
                    }
                )
                logger.info(f"Visualization uploaded to cloud: {result['url']}")
                
                # Create sharing link
                share_url = self.cloud_service.create_sharing_link(result['url'])
                logger.info(f"Sharing link created: {share_url}")
                
                # Store cloud info with insights
                self._insights.append("\nCloud Storage Information")
                self._insights.append("=======================")
                self._insights.append(f"Provider: {result['provider']}")
                self._insights.append(f"URL: {share_url}")
                self._insights.append(f"Timestamp: {result['timestamp']}")
                
            except Exception as e:
                logger.warning(f"Failed to upload to cloud: {e}")
                
        return output_file

    def _render_static(
        self,
        data: Dict[str, Any],
        output_path: Path,
        title: Optional[str] = None,
        figure_size: tuple = (15, 10)
    ) -> Path:
        """Render static visualization using matplotlib."""
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
        
        # Advanced correlation analysis
        if len(patterns) > 1:
            ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
            terms = list(patterns.keys())
            n_terms = len(terms)
            corr_matrix = np.zeros((n_terms, n_terms))
            
            # Calculate both Pearson and Spearman correlations
            pearson_matrix = np.zeros((n_terms, n_terms))
            spearman_matrix = np.zeros((n_terms, n_terms))
            
            for i, term1 in enumerate(terms):
                for j, term2 in enumerate(terms):
                    if i <= j:
                        values1 = patterns[term1]['values']
                        values2 = patterns[term2]['values']
                        pearson_corr, _ = pearsonr(values1, values2)
                        spearman_corr, _ = spearmanr(values1, values2)
                        
                        pearson_matrix[i,j] = pearson_matrix[j,i] = pearson_corr
                        spearman_matrix[i,j] = spearman_matrix[j,i] = spearman_corr
                        
                        # Use average of Pearson and Spearman for display
                        corr_matrix[i,j] = corr_matrix[j,i] = (pearson_corr + spearman_corr) / 2
            
            # Plot correlation heatmap
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
            ax2.set_title("Term Correlations\n(Pearson + Spearman avg)")
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

    def _render_interactive(
        self,
        data: Dict[str, Any],
        output_path: Path,
        title: Optional[str] = None
    ) -> Path:
        """Render interactive visualization using plotly."""
        time_points = data['time_points']
        patterns = data['patterns']
        metadata = data.get('metadata', {})
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"colspan": 2}, None],
                  [{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=(
                "Temporal Evolution",
                "Phase Space (2D)",
                "Phase Space (3D)"
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Main time series plot
        for term, pattern_data in patterns.items():
            values = pattern_data['values']
            meta = pattern_data['metadata']
            
            # Add main line
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=values,
                    name=term,
                    mode='lines',
                    hovertemplate=(
                        f"<b>{term}</b><br>" +
                        "Time: %{x}<br>" +
                        "Value: %{y:.3f}<br>" +
                        f"Trend: {meta['trend']}<br>" +
                        f"Volatility: {meta['volatility']:.3f}"
                    )
                ),
                row=1, col=1
            )
            
            # Add confidence interval if available
            if 'confidence' in meta:
                confidence = meta['confidence']
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=values + confidence * meta['std'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=values - confidence * meta['std'],
                        mode='lines',
                        line=dict(width=0),
                        fillcolor='rgba(68, 68, 68, 0.2)',
                        fill='tonexty',
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
        
        # Phase space plots (if multiple terms)
        if len(patterns) > 1:
            terms = list(patterns.keys())
            
            # 2D phase space
            if len(terms) >= 2:
                values1 = patterns[terms[0]]['values']
                values2 = patterns[terms[1]]['values']
                
                fig.add_trace(
                    go.Scatter(
                        x=values1,
                        y=values2,
                        mode='lines+markers',
                        name=f"{terms[0]} vs {terms[1]}",
                        marker=dict(
                            size=8,
                            color=time_points,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(
                                title="Time"
                            )
                        ),
                        hovertemplate=(
                            f"{terms[0]}: %{{x:.3f}}<br>" +
                            f"{terms[1]}: %{{y:.3f}}<br>" +
                            "Time: %{marker.color}<extra></extra>"
                        )
                    )
                )
            
            # 3D phase space
            if len(terms) >= 3:
                values1 = patterns[terms[0]]['values']
                values2 = patterns[terms[1]]['values']
                values3 = patterns[terms[2]]['values']
                
                fig.add_trace(
                    go.Scatter3d(
                        x=values1,
                        y=values2,
                        z=values3,
                        mode='lines+markers',
                        name=f"{terms[0]} vs {terms[1]} vs {terms[2]}",
                        marker=dict(
                            size=6,
                            color=time_points,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(
                                title="Time"
                            )
                        ),
                        hovertemplate=(
                            f"{terms[0]}: %{{x:.3f}}<br>" +
                            f"{terms[1]}: %{{y:.3f}}<br>" +
                            f"{terms[2]}: %{{z:.3f}}<br>" +
                            "Time: %{marker.color}<extra></extra>"
                        ),
                        line=dict(color='darkblue', width=2)
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=title or f"Interactive Temporal Analysis ({metadata.get('timeframe', '')})",
            showlegend=True,
            hovermode='closest',
            template='plotly_white',
            height=1000
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Semantic Intensity", row=1, col=1)
        
        if len(patterns) > 1:
            fig.update_xaxes(title_text=terms[0], row=2, col=1)
            fig.update_yaxes(title_text=terms[1], row=2, col=1)
            
            if len(terms) >= 3:
                fig.update_scenes(
                    dict(
                        xaxis_title=terms[0],
                        yaxis_title=terms[1],
                        zaxis_title=terms[2]
                    ),
                    row=2, col=2
                )
        
        # Save visualization
        output_file = output_path / f"timeseries_{metadata.get('timeframe', 'default')}_interactive.html"
        fig.write_html(
            str(output_file),
            include_plotlyjs='cdn',
            full_html=True,
            include_mathjax='cdn'
        )
        
        return output_file
        
    def render_server(
        self,
        data: Dict[str, Any],
        server_url: str,
        endpoint: str = '/api/v1/visualize/timeseries',
        store_in_cloud: bool = False
    ) -> Dict[str, Any]:
        """Render time series visualization using server.
        
        Args:
            data: Visualization data
            server_url: Server URL
            endpoint: API endpoint
            store_in_cloud: Whether to store result in cloud storage
            
        Returns:
            Server response
        """
        # Add cloud storage preference to request
        request_data = {
            'data': data,
            'context': {
                'temporal_scale': data['metadata']['temporal']['scale'],
                'granularity': data['metadata']['interval'],
                'cyclic_nature': data['metadata']['pattern_type'] == 'cyclic',
                'store_in_cloud': store_in_cloud
            }
        }
        
        # Add cloud service info if available
        if store_in_cloud and self.cloud_service:
            request_data['cloud_config'] = {
                'provider': self.cloud_service.provider_name,
                'credentials': self.cloud_service.credentials
            }
        
        response = requests.post(
            f"{server_url}{endpoint}",
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Store cloud info with insights if available
        if 'cloud_storage' in result:
            self._insights.append("\nCloud Storage Information")
            self._insights.append("=======================")
            self._insights.append(f"Provider: {result['cloud_storage']['provider']}")
            self._insights.append(f"URL: {result['cloud_storage']['url']}")
            self._insights.append(f"Timestamp: {result['cloud_storage']['timestamp']}")
        
        return result
        
    def render_insights(
        self,
        data: Dict[str, Any],
        hexagram_data: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None,
        upload_to_cloud: bool = False
    ) -> Optional[str]:
        """Render temporal insights, optionally including I Ching analysis.
        
        Args:
            data: Visualization data
            hexagram_data: Optional I Ching analysis data
            output_path: Optional output path for insights file
            upload_to_cloud: Whether to upload insights to cloud storage
            
        Returns:
            Path to insights file or insights text
        """
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
        
        # Phase Space Analysis
        phase_space = self._analyze_phase_space_topology(
            data['patterns'],
            data['metadata']['pattern_type']
        )
        
        insights.append("Phase Space Analysis")
        insights.append("===================")
        insights.append(f"Dimensionality: {phase_space['dimensionality']}")
        insights.append(f"Pattern Type: {phase_space['pattern_type']}")
        insights.append(f"Phase Space Volume: {phase_space['volume']:.3f}")
        insights.append("\nTrajectory Characteristics:")
        for i, length in enumerate(phase_space['trajectory_lengths']):
            insights.append(f"  {list(data['patterns'].keys())[i]}: Length = {length:.3f}")
            
        if phase_space['pattern_type'] == 'cyclic':
            insights.append("\nFrequency Analysis:")
            for term in data['patterns'].keys():
                if f'{term}_frequency' in phase_space:
                    insights.append(f"  {term}: {phase_space[f'{term}_frequency']:.3f} Hz")
        elif phase_space['pattern_type'] == 'linear':
            insights.append("\nTrend Analysis:")
            for term in data['patterns'].keys():
                if f'{term}_trend' in phase_space:
                    insights.append(f"  {term}: Slope = {phase_space[f'{term}_trend']:.3f}")
        
        # Semantic Interpretation
        insights.append("\nSemantic Evolution")
        insights.append("==================")
        interpretation = phase_space['semantic_analysis']['interpretation']
        insights.append(interpretation)
        
        # Pattern analysis
        insights.append("\nPattern Analysis")
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
        
        # Enhanced correlation analysis
        if data.get('correlations'):
            insights.append("\nCorrelation Analysis")
            insights.append("===================")
            for pair, corr in data['correlations'].items():
                term1, term2 = pair.split('-')
                # Calculate both Pearson and Spearman correlations
                values1 = data['patterns'][term1]['values']
                values2 = data['patterns'][term2]['values']
                pearson_corr, p_value = pearsonr(values1, values2)
                spearman_corr, s_p_value = spearmanr(values1, values2)
                
                relationship = (
                    "strong positive" if corr > 0.7 else
                    "moderate positive" if corr > 0.3 else
                    "weak positive" if corr > 0 else
                    "strong negative" if corr < -0.7 else
                    "moderate negative" if corr < -0.3 else
                    "weak negative"
                )
                
                insights.append(f"\n{term1} ↔ {term2}:")
                insights.append(f"  Overall: {relationship} ({corr:.2f})")
                insights.append(f"  Pearson: {pearson_corr:.2f} (p={p_value:.3f})")
                insights.append(f"  Spearman: {spearman_corr:.2f} (p={s_p_value:.3f})")
                
                # Phase relationship if both terms have phase information
                meta1 = data['patterns'][term1]['metadata']
                meta2 = data['patterns'][term2]['metadata']
                if 'phase' in meta1 and 'phase' in meta2:
                    phase_diff = abs(meta1['phase'] - meta2['phase'])
                    insights.append(f"  Phase Difference: {phase_diff:.2f} radians")
        
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
            
            # Upload insights to cloud if requested
            if upload_to_cloud and self.cloud_service:
                try:
                    result = self.cloud_service.upload_visualization(
                        insight_file,
                        metadata={
                            'type': 'insights',
                            'temporal_scale': data['metadata']['temporal']['scale'],
                            'pattern_type': data['metadata']['pattern_type'],
                            'terms': list(data['patterns'].keys())
                        }
                    )
                    logger.info(f"Insights uploaded to cloud: {result['url']}")
                    
                    # Create sharing link
                    share_url = self.cloud_service.create_sharing_link(result['url'])
                    logger.info(f"Insights sharing link created: {share_url}")
                    
                    # Add cloud storage info to insights
                    insights.append("\nCloud Storage Information")
                    insights.append("=======================")
                    insights.append(f"Provider: {result['provider']}")
                    insights.append(f"URL: {share_url}")
                    insights.append(f"Timestamp: {result['timestamp']}")
                    
                    # Update insight file with cloud info
                    insight_text = "\n".join(insights)
                    insight_file.write_text(insight_text)
                    
                except Exception as e:
                    logger.warning(f"Failed to upload insights to cloud: {e}")
            
            return str(insight_file)
            
        return insight_text
        
    def get_insights(self) -> List[str]:
        """Get generated insights from visualization."""
        return self._insights.copy()
        
    def clear_insights(self) -> None:
        """Clear stored insights."""
        self._insights = [] 