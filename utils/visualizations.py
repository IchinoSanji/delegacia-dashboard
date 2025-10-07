"""
Módulo de visualizações para o dashboard de risco operacional policial.
Foca em mapas, gráficos e visualizações específicas para segurança policial.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_risk_heatmap(df, x_col='hora', y_col='dia_semana', value_col='risco_alto'):
    """
    Cria mapa de calor de risco por tempo.
    
    Args:
        df (pandas.DataFrame): Dados de entrada
        x_col (str): Coluna para eixo X
        y_col (str): Coluna para eixo Y  
        value_col (str): Coluna de valor (risco)
        
    Returns:
        plotly.graph_objects.Figure: Figura do heatmap
    """
    try:
        # Verificar se as colunas existem
        if not all(col in df.columns for col in [x_col, y_col, value_col]):
            # Retornar figura vazia se colunas não existem
            fig = go.Figure()
            fig.add_annotation(text="Dados insuficientes para heatmap", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Criar pivot table para o heatmap
        pivot_data = df.groupby([y_col, x_col])[value_col].mean().reset_index()
        pivot_table = pivot_data.pivot(index=y_col, columns=x_col, values=value_col)
        
        # Ordenar dias da semana
        if y_col == 'dia_semana':
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_order_pt = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
            
            # Reordenar índice se possível
            available_days = [day for day in day_order if day in pivot_table.index]
            if available_days:
                pivot_table = pivot_table.reindex(available_days)
        
        # Criar heatmap
        fig = px.imshow(
            pivot_table,
            title=f'Mapa de Calor: Taxa de Risco por {y_col.replace("_", " ").title()} e {x_col.title()}',
            color_continuous_scale='Reds',
            aspect='auto',
            labels={'color': 'Taxa de Risco'}
        )
        
        # Customizar layout
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            height=500
        )
        
        return fig
        
    except Exception as e:
        # Retornar figura com mensagem de erro
        fig = go.Figure()
        fig.add_annotation(text=f"Erro ao criar heatmap: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_geographic_risk_map(df):
    """
    Cria mapa geográfico com zonas de risco.
    
    Args:
        df (pandas.DataFrame): Dados com latitude, longitude e risco
        
    Returns:
        plotly.graph_objects.Figure: Mapa geográfico
    """
    try:
        # Verificar se dados geográficos estão disponíveis
        geo_cols = ['latitude', 'longitude']
        if not all(col in df.columns for col in geo_cols):
            fig = go.Figure()
            fig.add_annotation(text="Dados geográficos não disponíveis", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Filtrar dados válidos
        valid_geo = df.dropna(subset=geo_cols)
        if len(valid_geo) == 0:
            fig = go.Figure()
            fig.add_annotation(text="Coordenadas geográficas inválidas", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Separar por nível de risco
        alto_risco = valid_geo[valid_geo['risco_alto'] == 1]
        baixo_risco = valid_geo[valid_geo['risco_alto'] == 0].sample(
            min(1000, len(valid_geo[valid_geo['risco_alto'] == 0])), 
            random_state=42
        )
        
        fig = go.Figure()
        
        # Adicionar pontos de baixo risco
        if len(baixo_risco) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=baixo_risco['latitude'],
                lon=baixo_risco['longitude'],
                mode='markers',
                marker=dict(
                    size=6,
                    color='green',
                    opacity=0.6
                ),
                name='Baixo Risco',
                hovertemplate='<b>%{text}</b><br>Risco: Baixo<extra></extra>',
                text=baixo_risco['bairro'] if 'bairro' in baixo_risco.columns else 'Local'
            ))
        
        # Adicionar pontos de alto risco
        if len(alto_risco) > 0:
            # Adicionar informações de arma ao hover
            hover_text = []
            for _, row in alto_risco.iterrows():
                text = f"Bairro: {row.get('bairro', 'N/A')}<br>"
                text += f"Crime: {row.get('tipo_crime', 'N/A')}<br>"
                text += f"Arma: {row.get('arma_utilizada', 'N/A')}"
                hover_text.append(text)
            
            fig.add_trace(go.Scattermapbox(
                lat=alto_risco['latitude'],
                lon=alto_risco['longitude'],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    opacity=0.8,
                    symbol='circle'
                ),
                name='Alto Risco ⚠️',
                hovertemplate='<b>ALTO RISCO</b><br>%{text}<extra></extra>',
                text=hover_text
            ))
        
        # Configurar layout do mapa
        center_lat = valid_geo['latitude'].mean()
        center_lon = valid_geo['longitude'].mean()
        
        fig.update_layout(
            title="🗺️ Mapa de Risco Operacional - Onde Patrulhar com Cautela",
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11
            ),
            height=600,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro ao criar mapa: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_temporal_analysis(df):
    """
    Cria análise temporal de risco com múltiplos gráficos.
    
    Args:
        df (pandas.DataFrame): Dados de entrada
        
    Returns:
        plotly.graph_objects.Figure: Figura com subplots
    """
    try:
        # Verificar colunas necessárias
        if 'risco_alto' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="Coluna 'risco_alto' não encontrada", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Criar subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risco por Hora do Dia', 'Risco por Dia da Semana', 
                          'Risco por Turno', 'Evolução Temporal'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # 1. Risco por hora
        if 'hora' in df.columns:
            hour_risk = df.groupby('hora')['risco_alto'].mean().reset_index()
            hour_risk['risco_pct'] = hour_risk['risco_alto'] * 100
            
            fig.add_trace(
                go.Bar(
                    x=hour_risk['hora'], 
                    y=hour_risk['risco_pct'],
                    name='Por Hora',
                    marker_color='red',
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # 2. Risco por dia da semana
        if 'dia_semana' in df.columns:
            day_risk = df.groupby('dia_semana')['risco_alto'].mean().reset_index()
            day_risk['risco_pct'] = day_risk['risco_alto'] * 100
            
            fig.add_trace(
                go.Bar(
                    x=day_risk['dia_semana'], 
                    y=day_risk['risco_pct'],
                    name='Por Dia',
                    marker_color='orange',
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # 3. Risco por turno
        if 'turno' in df.columns:
            turno_risk = df.groupby('turno').size().reset_index(name='count')
            
            fig.add_trace(
                go.Pie(
                    labels=turno_risk['turno'],
                    values=turno_risk['count'],
                    name="Por Turno"
                ),
                row=2, col=1
            )
        
        # 4. Evolução temporal (se disponível)
        if 'data_ocorrencia' in df.columns:
            # Agrupar por data
            df['data'] = df['data_ocorrencia'].dt.date
            temporal = df.groupby('data')['risco_alto'].mean().reset_index()
            temporal['risco_pct'] = temporal['risco_alto'] * 100
            
            # Pegar últimos 30 dias para não sobrecarregar
            temporal = temporal.tail(30)
            
            fig.add_trace(
                go.Scatter(
                    x=temporal['data'],
                    y=temporal['risco_pct'],
                    mode='lines+markers',
                    name='Evolução',
                    line=dict(color='blue')
                ),
                row=2, col=2
            )
        
        # Atualizar layout
        fig.update_layout(
            title_text="📊 Análise Temporal de Risco Operacional",
            height=600,
            showlegend=False
        )
        
        # Atualizar eixos Y para mostrar percentuais
        fig.update_yaxes(title_text="Taxa de Risco (%)", row=1, col=1)
        fig.update_yaxes(title_text="Taxa de Risco (%)", row=1, col=2)
        fig.update_yaxes(title_text="Taxa de Risco (%)", row=2, col=2)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro na análise temporal: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_feature_importance_chart(importance_df, top_n=10):
    """
    Cria gráfico de importância de features.
    
    Args:
        importance_df (pandas.DataFrame): DataFrame com features e importâncias
        top_n (int): Número de features para mostrar
        
    Returns:
        plotly.graph_objects.Figure: Gráfico de barras horizontais
    """
    try:
        if importance_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="Dados de importância não disponíveis", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Pegar top N features
        top_features = importance_df.head(top_n)
        
        # Criar gráfico de barras horizontais
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Fatores que Aumentam o Risco ao Policial',
            color='importance',
            color_continuous_scale='Reds'
        )
        
        # Customizar layout
        fig.update_layout(
            xaxis_title='Importância',
            yaxis_title='Fator de Risco',
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro no gráfico de importância: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_risk_distribution_chart(df):
    """
    Cria gráfico de distribuição de scores de risco.
    
    Args:
        df (pandas.DataFrame): Dados com scores de risco
        
    Returns:
        plotly.graph_objects.Figure: Histograma de distribuição
    """
    try:
        if 'risk_score' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="Scores de risco não disponíveis", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Criar histograma
        fig = px.histogram(
            df, 
            x='risk_score',
            nbins=30,
            title='Distribuição de Scores de Risco Operacional',
            color_discrete_sequence=['red']
        )
        
        # Adicionar linha vertical no percentil 80 (alto risco)
        threshold_80 = df['risk_score'].quantile(0.8)
        fig.add_vline(
            x=threshold_80, 
            line_dash="dash", 
            line_color="orange",
            annotation_text="Top 20% (Alto Risco)"
        )
        
        fig.update_layout(
            xaxis_title='Score de Risco',
            yaxis_title='Número de Ocorrências',
            height=400
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro na distribuição: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_weapon_analysis_chart(df):
    """
    Cria análise de tipos de armas utilizadas.
    
    Args:
        df (pandas.DataFrame): Dados com tipos de armas
        
    Returns:
        plotly.graph_objects.Figure: Gráfico de barras
    """
    try:
        if 'arma_utilizada' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="Dados de armas não disponíveis", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Contar tipos de armas
        weapon_counts = df['arma_utilizada'].value_counts().head(8)
        
        # Destacar armas mais perigosas
        colors = ['red' if weapon in ['Arma de Fogo', 'Explosivos'] else 'orange' 
                 for weapon in weapon_counts.index]
        
        fig = go.Figure(data=[
            go.Bar(
                x=weapon_counts.index,
                y=weapon_counts.values,
                marker_color=colors
            )
        ])
        
        fig.update_layout(
            title='Tipos de Armas Utilizadas em Ocorrências',
            xaxis_title='Tipo de Arma',
            yaxis_title='Número de Ocorrências',
            height=400
        )
        
        # Rotacionar labels do eixo X
        fig.update_xaxes(tickangle=45)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro na análise de armas: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_comparative_metrics_chart(metrics_dict):
    """
    Cria gráfico comparativo de métricas de performance.
    
    Args:
        metrics_dict (dict): Dicionário com métricas
        
    Returns:
        plotly.graph_objects.Figure: Gráfico de barras
    """
    try:
        if not metrics_dict:
            fig = go.Figure()
            fig.add_annotation(text="Métricas não disponíveis", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        metrics = ['precision', 'recall', 'f1_score']
        values = [metrics_dict.get(metric, 0) for metric in metrics]
        labels = ['Precisão', 'Recall', 'F1-Score']
        
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=['blue', 'green', 'red']
            )
        ])
        
        fig.update_layout(
            title='Métricas de Performance do Modelo',
            xaxis_title='Métrica',
            yaxis_title='Score',
            height=400,
            yaxis_range=[0, 1]
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro nas métricas: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_cluster_visualization(df_clustered, cluster_stats):
    """
    Cria visualização PCA dos clusters descobertos.
    
    Args:
        df_clustered (pandas.DataFrame): Dados com clusters e coordenadas PCA
        cluster_stats (pandas.DataFrame): Estatísticas dos clusters
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot dos clusters
    """
    try:
        if 'pca_1' not in df_clustered.columns or 'pca_2' not in df_clustered.columns:
            fig = go.Figure()
            fig.add_annotation(text="Dados PCA não disponíveis", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Criar scatter plot colorido por cluster
        fig = px.scatter(
            df_clustered,
            x='pca_1',
            y='pca_2',
            color='cluster',
            title='Clusters de Risco Operacional (Visualização PCA)',
            labels={'pca_1': 'Componente Principal 1', 'pca_2': 'Componente Principal 2'},
            color_continuous_scale='Viridis',
            opacity=0.6
        )
        
        # Adicionar centros dos clusters se disponível
        cluster_centers = df_clustered.groupby('cluster')[['pca_1', 'pca_2']].mean()
        
        fig.add_trace(go.Scatter(
            x=cluster_centers['pca_1'],
            y=cluster_centers['pca_2'],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='x',
                line=dict(width=2, color='black')
            ),
            name='Centros dos Clusters',
            showlegend=True
        ))
        
        fig.update_layout(height=500)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro na visualização: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_anomaly_visualization(df_anomalies):
    """
    Cria visualização de anomalias detectadas.
    
    Args:
        df_anomalies (pandas.DataFrame): Dados com flags de anomalias
        
    Returns:
        plotly.graph_objects.Figure: Visualização das anomalias
    """
    try:
        if 'anomaly_score' not in df_anomalies.columns:
            fig = go.Figure()
            fig.add_annotation(text="Scores de anomalia não disponíveis", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Separar normais e anomalias
        normal = df_anomalies[df_anomalies['is_anomaly'] == 0]
        anomalies = df_anomalies[df_anomalies['is_anomaly'] == 1]
        
        fig = go.Figure()
        
        # Adicionar distribuição de scores
        fig.add_trace(go.Histogram(
            x=normal['anomaly_score'],
            name='Ocorrências Normais',
            opacity=0.7,
            marker_color='green',
            nbinsx=30
        ))
        
        fig.add_trace(go.Histogram(
            x=anomalies['anomaly_score'],
            name='Anomalias Detectadas',
            opacity=0.7,
            marker_color='red',
            nbinsx=30
        ))
        
        fig.update_layout(
            title='Distribuição de Scores de Anomalia',
            xaxis_title='Score de Anomalia (maior = mais anômalo)',
            yaxis_title='Número de Ocorrências',
            height=400,
            barmode='overlay'
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro na visualização: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_elbow_plot(elbow_results):
    """
    Cria gráfico do método do cotovelo para determinar número ótimo de clusters.
    
    Args:
        elbow_results (dict): Resultados da análise do cotovelo
        
    Returns:
        plotly.graph_objects.Figure: Gráfico do cotovelo
    """
    try:
        if not elbow_results or 'k_range' not in elbow_results:
            fig = go.Figure()
            fig.add_annotation(text="Dados do método do cotovelo não disponíveis", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Criar subplot com múltiplas métricas
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Método do Cotovelo (Inércia)', 
                          'Silhouette Score', 
                          'Davies-Bouldin Score'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
        )
        
        k_range = elbow_results['k_range']
        
        # 1. Inércia (método do cotovelo)
        fig.add_trace(
            go.Scatter(
                x=k_range,
                y=elbow_results['inertias'],
                mode='lines+markers',
                name='Inércia',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Marcar k ótimo
        if 'optimal_k' in elbow_results:
            optimal_idx = k_range.index(elbow_results['optimal_k'])
            fig.add_trace(
                go.Scatter(
                    x=[elbow_results['optimal_k']],
                    y=[elbow_results['inertias'][optimal_idx]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name=f'Ótimo (k={elbow_results["optimal_k"]})',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. Silhouette score
        fig.add_trace(
            go.Scatter(
                x=k_range,
                y=elbow_results['silhouettes'],
                mode='lines+markers',
                name='Silhouette',
                line=dict(color='green', width=2),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # 3. Davies-Bouldin score (menor é melhor)
        fig.add_trace(
            go.Scatter(
                x=k_range,
                y=elbow_results['davies_bouldin'],
                mode='lines+markers',
                name='Davies-Bouldin',
                line=dict(color='orange', width=2),
                marker=dict(size=8)
            ),
            row=1, col=3
        )
        
        fig.update_xaxes(title_text="Número de Clusters (k)", row=1, col=1)
        fig.update_xaxes(title_text="Número de Clusters (k)", row=1, col=2)
        fig.update_xaxes(title_text="Número de Clusters (k)", row=1, col=3)
        
        fig.update_yaxes(title_text="Inércia", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=1, col=3)
        
        fig.update_layout(
            title_text="Determinação do Número Ótimo de Clusters",
            height=400,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro no gráfico do cotovelo: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_tsne_visualization(df_tsne):
    """
    Cria visualização t-SNE dos dados.
    
    Args:
        df_tsne (pandas.DataFrame): Dados com coordenadas t-SNE
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot t-SNE
    """
    try:
        if 'tsne_1' not in df_tsne.columns or 'tsne_2' not in df_tsne.columns:
            fig = go.Figure()
            fig.add_annotation(text="Dados t-SNE não disponíveis", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Criar scatter plot colorido por risco
        color_col = 'risco_alto' if 'risco_alto' in df_tsne.columns else None
        
        if color_col:
            fig = px.scatter(
                df_tsne,
                x='tsne_1',
                y='tsne_2',
                color=color_col,
                title='Visualização t-SNE: Padrões de Risco Operacional',
                labels={'tsne_1': 't-SNE Dimensão 1', 'tsne_2': 't-SNE Dimensão 2'},
                color_continuous_scale='RdYlGn_r',
                opacity=0.6
            )
        else:
            fig = px.scatter(
                df_tsne,
                x='tsne_1',
                y='tsne_2',
                title='Visualização t-SNE: Padrões de Risco Operacional',
                labels={'tsne_1': 't-SNE Dimensão 1', 'tsne_2': 't-SNE Dimensão 2'},
                opacity=0.6
            )
        
        fig.update_layout(height=500)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro na visualização t-SNE: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_pca_scree_plot(pca_results):
    """
    Cria scree plot para análise de componentes principais.
    
    Args:
        pca_results (dict): Resultados da análise PCA
        
    Returns:
        plotly.graph_objects.Figure: Scree plot
    """
    try:
        if not pca_results or 'explained_variance' not in pca_results:
            fig = go.Figure()
            fig.add_annotation(text="Dados PCA não disponíveis", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        n_components = len(pca_results['explained_variance'])
        components = list(range(1, n_components + 1))
        
        fig = go.Figure()
        
        # Variância explicada por componente
        fig.add_trace(go.Bar(
            x=components,
            y=pca_results['explained_variance'] * 100,
            name='Variância Explicada',
            marker_color='steelblue'
        ))
        
        # Linha de variância acumulada
        fig.add_trace(go.Scatter(
            x=components,
            y=pca_results['cumulative_variance'] * 100,
            name='Variância Acumulada',
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        # Marcar linha de 90%
        fig.add_hline(
            y=90, 
            line_dash="dash", 
            line_color="green",
            annotation_text="90% de variância",
            yref='y2'
        )
        
        # Marcar número de componentes para 90%
        if 'n_components_90' in pca_results:
            fig.add_vline(
                x=pca_results['n_components_90'],
                line_dash="dash",
                line_color="orange",
                annotation_text=f"PC{pca_results['n_components_90']}"
            )
        
        fig.update_layout(
            title='Scree Plot: Análise de Componentes Principais',
            xaxis_title='Componente Principal',
            yaxis_title='Variância Explicada (%)',
            yaxis2=dict(
                title='Variância Acumulada (%)',
                overlaying='y',
                side='right',
                range=[0, 105]
            ),
            height=400,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro no scree plot: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_association_rules_viz(rules_df):
    """
    Cria visualização de regras de associação.
    
    Args:
        rules_df (pandas.DataFrame): DataFrame com regras de associação
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot de regras
    """
    try:
        if rules_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="Nenhuma regra de associação encontrada", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Converter frozensets para strings
        rules_df = rules_df.copy()
        rules_df['antecedents_str'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_df['consequents_str'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_df['rule'] = rules_df['antecedents_str'] + ' → ' + rules_df['consequents_str']
        
        # Top 20 regras por lift
        top_rules = rules_df.nlargest(20, 'lift')
        
        # Criar scatter plot
        fig = px.scatter(
            top_rules,
            x='support',
            y='confidence',
            size='lift',
            color='lift',
            hover_data=['rule'],
            title='Top 20 Regras de Associação de Alto Risco',
            labels={
                'support': 'Suporte (Frequência)',
                'confidence': 'Confiança',
                'lift': 'Lift'
            },
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title='Suporte',
            yaxis_title='Confiança'
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro na visualização de regras: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def create_pattern_network(patterns_data):
    """
    Cria visualização de padrões de associação.
    
    Args:
        patterns_data (dict): Dados de padrões descobertos
        
    Returns:
        plotly.graph_objects.Figure: Gráfico de padrões
    """
    try:
        if not patterns_data or 'patterns' not in patterns_data or len(patterns_data['patterns']) == 0:
            fig = go.Figure()
            fig.add_annotation(text="Nenhum padrão significativo encontrado", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Pegar primeiro padrão para visualização
        first_pattern = patterns_data['patterns'][0]
        data = first_pattern['data']
        
        # Criar gráfico de barras para o padrão
        if 'taxa_risco' in data.columns:
            # Criar label descritivo
            if 'bairro' in data.columns and 'turno' in data.columns:
                data['label'] = data['bairro'] + ' - ' + data['turno']
            elif 'tipo_crime' in data.columns and 'arma_utilizada' in data.columns:
                data['label'] = data['tipo_crime'] + ' + ' + data['arma_utilizada']
            else:
                data['label'] = data.index.astype(str)
            
            fig = px.bar(
                data.head(10),
                x='label',
                y='taxa_risco',
                title=first_pattern['type'],
                labels={'taxa_risco': 'Taxa de Risco', 'label': ''},
                color='taxa_risco',
                color_continuous_scale='Reds'
            )
            
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=400)
            
        else:
            # Gráfico de contagem simples
            fig = px.bar(
                data.head(10),
                x=data.columns[0],
                y='count' if 'count' in data.columns else data.columns[-1],
                title=first_pattern['type']
            )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro nos padrões: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig
