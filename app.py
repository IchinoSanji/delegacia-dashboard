"""
🚓 Delegacia 5.0 — Dashboard de Risco Operacional para Segurança Policial

Foco: Proteção e segurança dos policiais em operações de campo
- Alertas visuais de risco por zona/turno
- Mapas de calor de perigo operacional  
- Recomendações táticas baseadas em ML
- Métricas de efetividade F1@20%
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# Importar utilitários
from utils.data_processor import DataProcessor
from utils.ml_models import RiskPredictor
from utils.visualizations import (
    create_risk_heatmap, create_geographic_risk_map, create_temporal_analysis,
    create_cluster_visualization, create_anomaly_visualization, create_pattern_network,
    create_elbow_plot, create_tsne_visualization, create_pca_scree_plot,
    create_association_rules_viz
)
from utils.risk_calculator import OfficerRiskCalculator
from utils.unsupervised_analysis import UnsupervisedRiskAnalyzer

# Configuração da página
st.set_page_config(
    page_title="🚓 Delegacia 5.0 - Risco Operacional",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS para indicadores de risco
st.markdown("""
<style>
.risk-high { 
    background-color: #FF4444; 
    color: white; 
    padding: 10px; 
    border-radius: 5px; 
    text-align: center;
    font-weight: bold;
}
.risk-medium { 
    background-color: #FFA500; 
    color: white; 
    padding: 10px; 
    border-radius: 5px; 
    text-align: center;
    font-weight: bold;
}
.risk-low { 
    background-color: #32CD32; 
    color: white; 
    padding: 10px; 
    border-radius: 5px; 
    text-align: center;
    font-weight: bold;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🚓 Delegacia 5.0 — Dashboard de Risco Operacional")
st.markdown("**Foco na Segurança do Policial:** Alertas, Mapas de Perigo e Recomendações Táticas")

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Inicializar processador de dados
@st.cache_data
def load_data():
    try:
        processor = DataProcessor("dataset_ocorrencias_delegacia_5.csv")
        return processor.get_processed_data()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None

df = load_data()

if df is None:
    st.stop()

# Filtros na sidebar
st.sidebar.subheader("🔍 Filtros")

# Filtro de período
if 'data_ocorrencia' in df.columns:
    min_date = df['data_ocorrencia'].min().date()
    max_date = df['data_ocorrencia'].max().date()
    
    date_range = st.sidebar.date_input(
        "Período de Análise",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[
            (df['data_ocorrencia'].dt.date >= start_date) & 
            (df['data_ocorrencia'].dt.date <= end_date)
        ].copy()
    else:
        df_filtered = df.copy()
else:
    df_filtered = df.copy()

# Filtros adicionais
bairros = ['Todos'] + sorted(df_filtered['bairro'].unique().tolist())
selected_bairro = st.sidebar.selectbox("Bairro", bairros)

if selected_bairro != 'Todos':
    df_filtered = df_filtered[df_filtered['bairro'] == selected_bairro]

# Novo filtro: Tipo de Crime
if 'tipo_crime' in df_filtered.columns:
    crimes = ['Todos'] + sorted(df_filtered['tipo_crime'].unique().tolist())
    selected_crime = st.sidebar.selectbox("Tipo de Crime", crimes)
    
    if selected_crime != 'Todos':
        df_filtered = df_filtered[df_filtered['tipo_crime'] == selected_crime]

# Novo filtro: Arma Utilizada
if 'arma_utilizada' in df_filtered.columns:
    armas = ['Todas'] + sorted(df_filtered['arma_utilizada'].unique().tolist())
    selected_arma = st.sidebar.selectbox("Tipo de Arma", armas)
    
    if selected_arma != 'Todas':
        df_filtered = df_filtered[df_filtered['arma_utilizada'] == selected_arma]

# Inicializar calculador de risco
risk_calc = OfficerRiskCalculator()
risk_predictor = RiskPredictor()

# Treinar modelo
if len(df_filtered) > 100:
    risk_predictor.train_model(df_filtered)

# Calcular riscos
df_filtered = risk_calc.calculate_risk_scores(df_filtered)

# Tabs principais
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚨 Alertas de Risco",
    "🗺️ Onde Patrulhar", 
    "⏰ Análise Temporal",
    "📊 Fatores de Risco",
    "📈 Efetividade Operacional",
    "🔬 Descoberta de Padrões"
])

# ================================
# TAB 1: ALERTAS DE RISCO
# ================================
with tab1:
    st.header("🚨 Sistema de Alertas de Risco Operacional")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Métricas principais
    total_ocorrencias = len(df_filtered)
    alto_risco = len(df_filtered[df_filtered['risco_alto'] == 1])
    taxa_risco = (alto_risco / total_ocorrencias * 100) if total_ocorrencias > 0 else 0
    
    with col1:
        st.metric("Total de Ocorrências", f"{total_ocorrencias:,}")
    
    with col2:
        st.metric("Situações de Alto Risco", f"{alto_risco:,}")
    
    with col3:
        st.metric("Taxa de Risco", f"{taxa_risco:.1f}%")
    
    with col4:
        # Status de alerta geral
        if taxa_risco >= 15:
            st.markdown('<div class="risk-high">⚠️ ALERTA MÁXIMO</div>', unsafe_allow_html=True)
        elif taxa_risco >= 10:
            st.markdown('<div class="risk-medium">🟡 ATENÇÃO</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">🟢 SITUAÇÃO NORMAL</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Alertas por bairro
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏘️ Risco por Bairro (Últimas 24h)")
        
        # Simular dados das últimas 24h (usar dados mais recentes)
        recent_data = df_filtered.tail(min(200, len(df_filtered)))
        bairro_risk = recent_data.groupby('bairro').agg({
            'risco_alto': ['count', 'sum', 'mean'],
            'risk_score': 'mean'
        }).round(2)
        
        bairro_risk.columns = ['Total', 'Alto_Risco', 'Taxa_Risco', 'Score_Medio']
        bairro_risk = bairro_risk.sort_values('Taxa_Risco', ascending=False)
        
        for bairro, row in bairro_risk.head(10).iterrows():
            taxa = row['Taxa_Risco'] * 100
            score = row['Score_Medio']
            
            if taxa >= 20:
                color_class = "risk-high"
                icon = "🔴"
            elif taxa >= 10:
                color_class = "risk-medium" 
                icon = "🟡"
            else:
                color_class = "risk-low"
                icon = "🟢"
            
            st.markdown(f"""
            <div class="{color_class}">
                {icon} <strong>{bairro}</strong><br>
                Taxa: {taxa:.1f}% | Score: {score:.2f}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("🕐 Risco por Turno Atual")
        
        # Análise por turno
        turno_risk = recent_data.groupby('turno').agg({
            'risco_alto': ['count', 'sum', 'mean'],
            'risk_score': 'mean'
        }).round(2)
        
        turno_risk.columns = ['Total', 'Alto_Risco', 'Taxa_Risco', 'Score_Medio']
        turno_risk = turno_risk.sort_values('Taxa_Risco', ascending=False)
        
        for turno, row in turno_risk.iterrows():
            taxa = row['Taxa_Risco'] * 100
            score = row['Score_Medio']
            
            if taxa >= 20:
                color_class = "risk-high"
                icon = "🔴"
            elif taxa >= 10:
                color_class = "risk-medium"
                icon = "🟡" 
            else:
                color_class = "risk-low"
                icon = "🟢"
            
            st.markdown(f"""
            <div class="{color_class}">
                {icon} <strong>{turno}</strong><br>
                Taxa: {taxa:.1f}% | Score: {score:.2f}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

# ================================
# TAB 2: ONDE PATRULHAR
# ================================
with tab2:
    st.header("🗺️ Mapa de Patrulhamento Tático")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔥 Zonas de Alto Risco para Policiais")
        
        # Criar mapa de calor geográfico
        if 'latitude' in df_filtered.columns and 'longitude' in df_filtered.columns:
            fig_map = create_geographic_risk_map(df_filtered)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Dados geográficos não disponíveis")
    
    with col2:
        st.subheader("🛡️ Recomendações Táticas")
        
        # Top 5 bairros mais perigosos
        top_dangerous = df_filtered.groupby('bairro').agg({
            'risco_alto': 'mean',
            'arma_utilizada': lambda x: (x.isin(['Arma de Fogo', 'Explosivos'])).sum()
        }).sort_values('risco_alto', ascending=False).head(5)
        
        st.markdown("**🎯 Prioridade Máxima:**")
        for bairro, data in top_dangerous.iterrows():
            risk_pct = data['risco_alto'] * 100
            weapons = data['arma_utilizada']
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>{bairro}</strong><br>
                Risco: {risk_pct:.1f}%<br>
                Armas: {weapons} casos<br>
                <em>👥 Requer patrulha dupla</em>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Análise de armas por bairro
    st.subheader("🔫 Distribuição de Armas por Zona")
    
    weapon_analysis = df_filtered.groupby(['bairro', 'arma_utilizada']).size().unstack(fill_value=0)
    
    if not weapon_analysis.empty:
        # Foco nas armas mais perigosas
        dangerous_weapons = ['Arma de Fogo', 'Explosivos']
        available_weapons = [w for w in dangerous_weapons if w in weapon_analysis.columns]
        
        if available_weapons:
            fig_weapons = px.bar(
                weapon_analysis[available_weapons].reset_index(),
                x='bairro',
                y=available_weapons,
                title="Ocorrências com Armas Letais por Bairro",
                color_discrete_sequence=['#FF4444', '#FF8888'],
                height=400
            )
            fig_weapons.update_xaxes(tickangle=45)
            st.plotly_chart(fig_weapons, use_container_width=True)

# ================================
# TAB 3: ANÁLISE TEMPORAL
# ================================
with tab3:
    st.header("⏰ Quando os Policiais Correm Mais Risco?")
    
    # Análise por hora e dia da semana
    temporal_fig = create_temporal_analysis(df_filtered)
    st.plotly_chart(temporal_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Risco por Dia da Semana")
        
        day_risk = df_filtered.groupby('dia_semana').agg({
            'risco_alto': ['count', 'sum', 'mean']
        })
        day_risk.columns = ['Total', 'Alto_Risco', 'Taxa_Risco']
        day_risk['Taxa_Risco_Pct'] = day_risk['Taxa_Risco'] * 100
        
        fig_days = px.bar(
            day_risk.reset_index(),
            x='dia_semana',
            y='Taxa_Risco_Pct',
            title="Taxa de Alto Risco por Dia da Semana (%)",
            color='Taxa_Risco_Pct',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_days, use_container_width=True)
    
    with col2:
        st.subheader("🕐 Risco por Turno")
        
        shift_risk = df_filtered.groupby('turno').agg({
            'risco_alto': ['count', 'sum', 'mean'],
            'arma_utilizada': lambda x: (x.isin(['Arma de Fogo', 'Explosivos'])).sum()
        })
        shift_risk.columns = ['Total', 'Alto_Risco', 'Taxa_Risco', 'Armas_Letais']
        shift_risk['Taxa_Risco_Pct'] = shift_risk['Taxa_Risco'] * 100
        
        fig_shifts = px.pie(
            shift_risk.reset_index(),
            values='Armas_Letais',
            names='turno',
            title="Distribuição de Armas Letais por Turno"
        )
        st.plotly_chart(fig_shifts, use_container_width=True)
    
    st.markdown("---")
    
    # Métricas operacionais por turno
    st.subheader("📊 Métricas Operacionais por Turno")
    
    metrics_df = df_filtered.groupby('turno').agg({
        'risco_alto': ['count', 'sum', 'mean'],
        'quantidade_suspeitos': 'mean',
        'quantidade_vitimas': 'mean'
    }).round(2)
    
    metrics_df.columns = ['Total_Ocorrencias', 'Alto_Risco', 'Taxa_Risco', 'Media_Suspeitos', 'Media_Vitimas']
    metrics_df['Taxa_Risco_Pct'] = (metrics_df['Taxa_Risco'] * 100).round(1)
    
    st.dataframe(
        metrics_df[['Total_Ocorrencias', 'Alto_Risco', 'Taxa_Risco_Pct', 'Media_Suspeitos', 'Media_Vitimas']],
        use_container_width=True
    )

# ================================
# TAB 4: FATORES DE RISCO
# ================================
with tab4:
    st.header("📊 O Que Torna Uma Situação Mais Perigosa?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Principais Fatores de Risco")
        
        # Análise de importância das features
        if hasattr(risk_predictor, 'model') and risk_predictor.model is not None:
            try:
                feature_importance = risk_predictor.get_feature_importance()
                
                if not feature_importance.empty:
                    fig_importance = px.bar(
                        feature_importance.head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Fatores que Aumentam o Risco ao Policial",
                        color='importance',
                        color_continuous_scale='Reds'
                    )
                    fig_importance.update_layout(height=500)
                    st.plotly_chart(fig_importance, use_container_width=True)
            except Exception as e:
                st.info("Análise de importância indisponível")
    
    with col2:
        st.subheader("🔍 Análise por Tipo de Crime")
        
        crime_risk = df_filtered.groupby('tipo_crime').agg({
            'risco_alto': ['count', 'sum', 'mean'],
            'quantidade_suspeitos': 'mean'
        }).round(2)
        
        crime_risk.columns = ['Total', 'Alto_Risco', 'Taxa_Risco', 'Media_Suspeitos']
        crime_risk['Taxa_Risco_Pct'] = crime_risk['Taxa_Risco'] * 100
        crime_risk = crime_risk.sort_values('Taxa_Risco', ascending=False)
        
        # Destacar os crimes mais perigosos
        st.markdown("**🚨 Crimes Mais Perigosos para Policiais:**")
        
        for crime, data in crime_risk.head(8).iterrows():
            taxa = data['Taxa_Risco_Pct']
            suspeitos = data['Media_Suspeitos']
            total = data['Total']
            
            if taxa >= 50:
                color = "🔴"
            elif taxa >= 25:
                color = "🟡"
            else:
                color = "🟢"
            
            st.markdown(f"""
            **{color} {crime}**  
            Taxa de Alto Risco: {taxa:.1f}%  
            Média de Suspeitos: {suspeitos:.1f}  
            Total de Casos: {total}
            """)
            st.markdown("---")
    
    # Matriz de correlação de fatores de risco
    st.subheader("🔗 Correlação entre Fatores de Risco")
    
    numeric_cols = ['risco_alto', 'quantidade_suspeitos', 'quantidade_vitimas', 'hora', 'dia_semana_idx']
    available_cols = [col for col in numeric_cols if col in df_filtered.columns]
    
    if len(available_cols) > 2:
        corr_matrix = df_filtered[available_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlação entre Variáveis de Risco",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ================================
# TAB 5: EFETIVIDADE OPERACIONAL
# ================================
with tab5:
    st.header("📈 Ganho Operacional com IA Preditiva")
    
    if hasattr(risk_predictor, 'model') and risk_predictor.model is not None:
        
        col1, col2, col3 = st.columns(3)
        
        # Calcular métricas F1@20%
        try:
            f1_20_results = risk_predictor.calculate_f1_at_k(df_filtered, k=0.20)
            precision_20 = f1_20_results.get('precision', 0)
            recall_20 = f1_20_results.get('recall', 0)
            f1_20 = f1_20_results.get('f1_score', 0)
            
            with col1:
                st.metric(
                    "F1-Score @20%", 
                    f"{f1_20:.3f}",
                    help="Efetividade ao priorizar apenas 20% das ocorrências"
                )
            
            with col2:
                st.metric(
                    "Precisão @20%",
                    f"{precision_20:.3f}",
                    help="% de situações de alto risco identificadas corretamente"
                )
            
            with col3:
                st.metric(
                    "Recall @20%",
                    f"{recall_20:.3f}",
                    help="% do total de situações de risco capturadas"
                )
        except Exception:
            st.info("Métricas de efetividade indisponíveis")
        
        st.markdown("---")
        
        # Simulação de ganho operacional
        st.subheader("🎯 Simulação: Priorizando 20% das Ocorrências")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📋 Cenário Atual (sem IA):**
            - Todas as ocorrências tratadas igualmente
            - Resposta reativa aos riscos
            - Alocação uniforme de recursos
            """)
            
            # Métricas baseline
            total_casos = len(df_filtered)
            casos_alto_risco = len(df_filtered[df_filtered['risco_alto'] == 1])
            
            st.markdown(f"""
            **📊 Números Atuais:**
            - Total de casos: {total_casos:,}
            - Casos de alto risco: {casos_alto_risco:,}
            - Taxa de risco: {(casos_alto_risco/total_casos*100):.1f}%
            """)
        
        with col2:
            st.markdown("""
            **🤖 Cenário com IA (F1@20%):**
            - Priorizacao inteligente das ocorrências
            - Foco nos 20% mais perigosos
            - Otimização de recursos policiais
            """)
            
            # Calcular benefícios
            casos_priorizados = int(total_casos * 0.20)
            casos_risco_capturados = int(casos_alto_risco * recall_20)
            
            st.markdown(f"""
            **🎯 Resultados Esperados:**
            - Casos priorizados: {casos_priorizados:,} (20%)
            - Alto risco capturado: {casos_risco_capturados:,}
            - Eficiência: {(casos_risco_capturados/casos_priorizados*100):.1f}%
            """)
        
        # Gráfico de performance
        st.subheader("📊 Curva de Performance do Modelo")
        
        if len(df_filtered) > 50:
            try:
                performance_data = risk_predictor.get_performance_curve(df_filtered)
                
                fig_perf = go.Figure()
                
                fig_perf.add_trace(go.Scatter(
                    x=performance_data['k_values'],
                    y=performance_data['precision'],
                    mode='lines+markers',
                    name='Precisão',
                    line=dict(color='blue')
                ))
                
                fig_perf.add_trace(go.Scatter(
                    x=performance_data['k_values'],
                    y=performance_data['recall'],  # Aqui estava o erro
                    mode='lines+markers',
                    name='Recall',
                    line=dict(color='red')
                ))
                
                fig_perf.add_trace(go.Scatter(
                    x=performance_data['k_values'],
                    y=performance_data['f1_score'],
                    mode='lines+markers',
                    name='F1-Score',
                    line=dict(color='green')
                ))
                
                fig_perf.add_vline(x=20, line_dash="dash", line_color="orange",
                                  annotation_text="F1@20%")
                
                fig_perf.update_layout(
                    title="Performance do Modelo por % de Ocorrências Priorizadas",
                    xaxis_title="% de Ocorrências Priorizadas",
                    yaxis_title="Score",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_perf, use_container_width=True)
                
            except Exception as e:
                st.info("Gráfico de performance indisponível")
    
    else:
        st.info("Modelo não treinado. Dados insuficientes.")
    
    st.markdown("---")
    
    # Relatório de Inteligência Operacional
    st.subheader("📋 Relatório de Inteligência Operacional")
    
    with st.expander("📊 Recomendações Automáticas", expanded=True):
        
        # Top bairros de risco
        top_risk_bairros = df_filtered.groupby('bairro')['risco_alto'].mean().sort_values(ascending=False).head(3)
        
        st.markdown("**🎯 Reforço Policial Recomendado:**")
        for bairro, risk_rate in top_risk_bairros.items():
            st.markdown(f"• **{bairro}**: Taxa de risco {risk_rate*100:.1f}% - Requer patrulhamento reforçado")
        
        # Horários críticos
        hour_risk = df_filtered.groupby('hora')['risco_alto'].mean().sort_values(ascending=False).head(3)
        
        st.markdown("**⏰ Horários Críticos:**")
        for hour, risk_rate in hour_risk.items():
            st.markdown(f"• **{hour:02d}h**: Taxa de risco {risk_rate*100:.1f}% - Aumentar efetivo")
        
        # Tipos de crime perigosos
        crime_risk = df_filtered.groupby('tipo_crime')['risco_alto'].mean().sort_values(ascending=False).head(3)
        
        st.markdown("**🚨 Protocolos Especiais:**")
        for crime, risk_rate in crime_risk.items():
            if risk_rate > 0.3:  # Acima de 30% de risco
                st.markdown(f"• **{crime}**: Protocolo de alta segurança - Backup obrigatório")

# ================================
# TAB 6: ANÁLISE NÃO SUPERVISIONADA
# ================================
with tab6:
    st.header("🔬 Descoberta de Padrões Ocultos (Análise Não Supervisionada)")
    
    st.markdown("""
    Esta aba utiliza **Machine Learning Não Supervisionado** para descobrir padrões que não são óbvios:
    - **Clustering:** Grupos naturais de ocorrências similares
    - **Detecção de Anomalias:** Situações atípicas que fogem do padrão
    - **Associações:** Combinações de fatores que elevam o risco
    """)
    
    # Verificar se há dados suficientes
    if len(df_filtered) < 50:
        st.warning("⚠️ Dados insuficientes para análise não supervisionada. Ajuste os filtros.")
        st.stop()
    
    # Inicializar analisador não supervisionado
    unsupervised = UnsupervisedRiskAnalyzer()
    
    # Subtabs expandidos
    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "🎯 Clustering de Risco",
        "📊 Análise PCA & t-SNE",
        "⚡ Detecção de Anomalias",
        "🔗 Regras de Associação",
        "📈 Padrões Combinados"
    ])
    
    # ---- SUBTAB 1: CLUSTERING ----
    with subtab1:
        st.subheader("🎯 Descoberta de Grupos de Risco")
        
        st.markdown("""
        **O que é?** O algoritmo agrupa ocorrências similares sem saber quais são "perigosas".  
        **Por quê?** Revela padrões naturais que podem indicar novos tipos de risco operacional.
        """)
        
        # Passo 1: Método do cotovelo para encontrar k ótimo
        with st.expander("📐 Passo 1: Determinar Número Ótimo de Clusters (Método do Cotovelo)", expanded=False):
            try:
                with st.spinner("Calculando número ótimo de clusters..."):
                    elbow_results = unsupervised.find_optimal_clusters(df_filtered, max_clusters=8)
                
                st.info(f"💡 **Número ótimo sugerido:** {elbow_results['optimal_k']} clusters (baseado na curvatura da inércia)")
                st.info(f"💡 **Melhor Silhouette Score:** k = {elbow_results['best_silhouette_k']}")
                
                # Visualização do método do cotovelo
                fig_elbow = create_elbow_plot(elbow_results)
                st.plotly_chart(fig_elbow, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Método do cotovelo indisponível: {str(e)}")
        
        st.markdown("---")
        
        # Passo 2: Executar clustering
        st.subheader("📊 Passo 2: Executar Clustering")
        
        # Controles
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            cluster_method = st.selectbox("Método de Clustering", ["K-Means", "DBSCAN"], index=0)
        with col2:
            if cluster_method == "K-Means":
                n_clusters = st.slider("Número de Clusters", min_value=2, max_value=8, value=4)
            else:
                st.info("DBSCAN encontra clusters automaticamente")
        
        try:
            method_param = 'kmeans' if cluster_method == "K-Means" else 'dbscan'
            
            with st.spinner(f"Executando {cluster_method} clustering..."):
                if method_param == 'kmeans':
                    clustering_results = unsupervised.perform_clustering(df_filtered, n_clusters=n_clusters, method='kmeans')
                else:
                    clustering_results = unsupervised.perform_clustering(df_filtered, method='dbscan')
            
            # Métricas de qualidade
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Silhouette Score",
                    f"{clustering_results['silhouette_score']:.3f}",
                    help="Qualidade do clustering (0-1, maior = melhor)"
                )
            
            with col2:
                st.metric(
                    "Clusters Encontrados",
                    clustering_results['n_clusters']
                )
            
            with col3:
                st.metric(
                    "Variância Explicada (PCA)",
                    f"{clustering_results['pca_variance_explained']*100:.1f}%",
                    help="% da variação dos dados capturada em 2D"
                )
            
            st.markdown("---")
            
            # Visualização dos clusters
            st.subheader("📊 Visualização dos Clusters (PCA)")
            fig_clusters = create_cluster_visualization(
                clustering_results['df_clustered'],
                clustering_results['cluster_stats']
            )
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            st.markdown("---")
            
            # Interpretação dos clusters
            st.subheader("🔍 Interpretação dos Clusters")
            
            cluster_interpretations = unsupervised.get_cluster_interpretation(
                clustering_results['cluster_stats']
            )
            
            for interp in cluster_interpretations:
                with st.expander(f"{interp['profile']} - Cluster {interp['cluster_id']}", expanded=True):
                    st.markdown(f"**Descrição:** {interp['description']}")
                    st.markdown(f"**Recomendação:** {interp['recommendation']}")
                    
                    st.markdown("**Características Típicas:**")
                    chars = interp['characteristics']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"- Crime mais comum: **{chars['crime_comum']}**")
                        st.markdown(f"- Bairro mais comum: **{chars['bairro_comum']}**")
                    with col2:
                        st.markdown(f"- Turno mais comum: **{chars['turno_comum']}**")
                        st.markdown(f"- Tamanho: **{chars['tamanho']}**")
            
            # Estatísticas detalhadas
            st.markdown("---")
            st.subheader("📋 Estatísticas Detalhadas por Cluster")
            
            stats_display = clustering_results['cluster_stats'].copy()
            stats_display['high_risk_rate'] = (stats_display['high_risk_rate'] * 100).round(1)
            stats_display['weapon_rate'] = (stats_display['weapon_rate'] * 100).round(1)
            stats_display['percentage'] = stats_display['percentage'].round(1)
            
            st.dataframe(
                stats_display[[
                    'cluster_id', 'size', 'percentage', 'high_risk_rate', 
                    'weapon_rate', 'avg_suspects', 'most_common_crime'
                ]].rename(columns={
                    'cluster_id': 'Cluster',
                    'size': 'Tamanho',
                    'percentage': '% do Total',
                    'high_risk_rate': 'Taxa Alto Risco (%)',
                    'weapon_rate': 'Taxa Arma Letal (%)',
                    'avg_suspects': 'Média Suspeitos',
                    'most_common_crime': 'Crime Mais Comum'
                }),
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Erro no clustering: {str(e)}")
    
    # ---- SUBTAB 2: ANÁLISE PCA & t-SNE ----
    with subtab2:
        st.subheader("📊 Redução Dimensional e Visualização Avançada")
        
        st.markdown("""
        **O que é?** Técnicas para reduzir dados multidimensionais em 2D, mantendo informações importantes.  
        **Por quê?** Permite visualizar padrões complexos de forma intuitiva.
        """)
        
        # PCA Completo
        st.markdown("### 🔬 Análise de Componentes Principais (PCA)")
        
        try:
            with st.spinner("Executando análise PCA completa..."):
                pca_results = unsupervised.perform_pca_analysis(df_filtered)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Componentes para 90% de Variância",
                    pca_results['n_components_90'],
                    help="Número de componentes necessários para capturar 90% da variação dos dados"
                )
            
            with col2:
                st.metric(
                    "Total de Componentes",
                    pca_results['n_components']
                )
            
            # Scree plot
            st.markdown("#### 📈 Scree Plot: Variância Explicada por Componente")
            fig_scree = create_pca_scree_plot(pca_results)
            st.plotly_chart(fig_scree, use_container_width=True)
            
            # Loadings (contribuição das features)
            with st.expander("📋 Contribuição das Features (Loadings)", expanded=False):
                st.markdown("**As features mais importantes em cada componente principal:**")
                
                # Mostrar top 3 componentes
                loadings_display = pca_results['loadings'].iloc[:, :min(3, pca_results['n_components'])]
                loadings_display = loadings_display.round(3)
                st.dataframe(loadings_display, use_container_width=True)
                
                st.info("""
                **Como interpretar:**
                - Valores positivos altos: Feature contribui positivamente para o componente
                - Valores negativos altos: Feature contribui negativamente
                - Valores próximos de 0: Feature tem pouca contribuição
                """)
            
        except Exception as e:
            st.warning(f"Análise PCA indisponível: {str(e)}")
        
        st.markdown("---")
        
        # t-SNE
        st.markdown("### 🎨 Visualização t-SNE")
        
        st.markdown("""
        **t-SNE** (t-Distributed Stochastic Neighbor Embedding) é uma técnica alternativa ao PCA 
        que frequentemente revela padrões locais e clusters de forma mais clara.
        """)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30, 
                                  help="Controla o balanço entre estrutura local e global")
        
        try:
            with st.spinner("Calculando t-SNE (pode levar alguns segundos)..."):
                tsne_results = unsupervised.perform_tsne(df_filtered, perplexity=perplexity)
            
            st.success(f"✅ t-SNE executado com perplexity = {tsne_results['perplexity_used']}")
            
            # Visualização t-SNE
            fig_tsne = create_tsne_visualization(tsne_results['df_tsne'])
            st.plotly_chart(fig_tsne, use_container_width=True)
            
            st.info("""
            **💡 Dica:** No t-SNE, pontos próximos indicam ocorrências similares. 
            Procure por:
            - **Clusters separados**: Grupos distintos de risco
            - **Gradientes de cor**: Transições de baixo para alto risco
            - **Outliers**: Pontos isolados (potenciais anomalias)
            """)
            
        except Exception as e:
            st.warning(f"t-SNE indisponível: {str(e)}")
    
    # ---- SUBTAB 3: ANOMALIAS ----
    with subtab3:
        st.subheader("⚡ Detecção de Ocorrências Anômalas")
        
        st.markdown("""
        **O que é?** Identifica ocorrências que "não se encaixam" no padrão normal.  
        **Por quê?** Anomalias podem ser situações raras mas extremamente perigosas, ou indicar mudanças no perfil criminal.
        """)
        
        # Controle de sensibilidade
        col1, col2 = st.columns([3, 1])
        with col2:
            contamination = st.slider(
                "Sensibilidade (%)",
                min_value=5,
                max_value=20,
                value=10,
                help="% de ocorrências esperadas como anômalas"
            ) / 100
        
        try:
            with st.spinner("Detectando anomalias com Isolation Forest..."):
                anomaly_results = unsupervised.detect_anomalies(df_filtered, contamination=contamination)
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            
            stats = anomaly_results['stats']
            
            with col1:
                st.metric(
                    "Anomalias Detectadas",
                    stats['total_anomalies'],
                    f"{stats['anomaly_percentage']:.1f}% do total"
                )
            
            with col2:
                st.metric(
                    "Taxa de Risco (Anomalias)",
                    f"{stats['anomalies_high_risk_rate']*100:.1f}%",
                    help="% de anomalias que são alto risco"
                )
            
            with col3:
                st.metric(
                    "Taxa de Risco (Normais)",
                    f"{stats['normal_high_risk_rate']*100:.1f}%",
                    help="% de ocorrências normais que são alto risco"
                )
            
            # Comparação
            if stats['anomalies_high_risk_rate'] > stats['normal_high_risk_rate']:
                diff = (stats['anomalies_high_risk_rate'] - stats['normal_high_risk_rate']) * 100
                st.info(f"✅ **Insight:** Anomalias têm {diff:.1f}% mais chance de serem alto risco que ocorrências normais!")
            
            st.markdown("---")
            
            # Visualização
            st.subheader("📊 Distribuição de Scores de Anomalia")
            fig_anomalies = create_anomaly_visualization(anomaly_results['df_anomalies'])
            st.plotly_chart(fig_anomalies, use_container_width=True)
            
            st.markdown("---")
            
            # Top anomalias
            st.subheader("🔥 Top 10 Ocorrências Mais Anômalas")
            
            top_anomalies = anomaly_results['top_anomalies'].head(10)
            
            for idx, (_, row) in enumerate(top_anomalies.iterrows(), 1):
                with st.expander(f"#{idx} - Anomalia Score: {row['anomaly_score']:.3f}", expanded=(idx <= 3)):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Bairro:** {row.get('bairro', 'N/A')}")
                        st.markdown(f"**Crime:** {row.get('tipo_crime', 'N/A')}")
                        st.markdown(f"**Arma:** {row.get('arma_utilizada', 'N/A')}")
                    
                    with col2:
                        st.markdown(f"**Turno:** {row.get('turno', 'N/A')}")
                        st.markdown(f"**Suspeitos:** {row.get('quantidade_suspeitos', 'N/A')}")
                        st.markdown(f"**Vítimas:** {row.get('quantidade_vitimas', 'N/A')}")
                    
                    if row.get('risco_alto', 0) == 1:
                        st.error("🔴 **ALTO RISCO CONFIRMADO**")
                    else:
                        st.warning("🟡 Padrão atípico - investigar")
            
            # Análise de locais
            if stats['top_anomaly_locations']:
                st.markdown("---")
                st.subheader("📍 Bairros com Mais Anomalias")
                
                locations_df = pd.DataFrame(
                    list(stats['top_anomaly_locations'].items()),
                    columns=['Bairro', 'Número de Anomalias']
                )
                
                fig_locations = px.bar(
                    locations_df,
                    x='Bairro',
                    y='Número de Anomalias',
                    title="Distribuição Geográfica de Anomalias",
                    color='Número de Anomalias',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_locations, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro na detecção de anomalias: {str(e)}")
    
    # ---- SUBTAB 4: REGRAS DE ASSOCIAÇÃO ----
    with subtab4:
        st.subheader("🔗 Mineração de Regras de Associação (Apriori)")
        
        st.markdown("""
        **O que é?** Algoritmo Apriori descobre regras do tipo "SE X E Y ENTÃO Z" em situações de alto risco.  
        **Por quê?** Revela combinações específicas de fatores que elevam drasticamente o risco operacional.
        """)
        
        # Controles
        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider("Suporte Mínimo (%)", min_value=1, max_value=20, value=5,
                                   help="Frequência mínima da combinação") / 100
        with col2:
            min_confidence = st.slider("Confiança Mínima (%)", min_value=30, max_value=100, value=50,
                                      help="Força da regra (probabilidade condicional)") / 100
        
        try:
            with st.spinner("Minerando regras de associação com Apriori..."):
                association_results = unsupervised.find_association_rules(
                    df_filtered, 
                    min_support=min_support,
                    min_confidence=min_confidence
                )
            
            if association_results['total_rules'] == 0:
                st.info("⚠️ Nenhuma regra encontrada com esses parâmetros. Tente reduzir o suporte ou confiança mínimos.")
            else:
                st.success(f"✅ {association_results['total_rules']} regras de associação descobertas!")
                
                # Visualização das regras
                st.markdown("### 📊 Visualização de Regras (Support × Confidence × Lift)")
                fig_rules = create_association_rules_viz(association_results['rules'])
                st.plotly_chart(fig_rules, use_container_width=True)
                
                st.markdown("---")
                
                # Top regras
                st.markdown("### 🔥 Top 10 Regras Mais Importantes (por Lift)")
                
                top_rules = association_results['rules'].nlargest(10, 'lift')
                
                for idx, (_, rule) in enumerate(top_rules.iterrows(), 1):
                    # Converter frozensets para strings legíveis
                    antecedents = ', '.join([item.replace('_', ' ') for item in list(rule['antecedents'])])
                    consequents = ', '.join([item.replace('_', ' ') for item in list(rule['consequents'])])
                    
                    with st.expander(f"#{idx}: {antecedents} → {consequents} (Lift: {rule['lift']:.2f})", expanded=(idx <= 3)):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Suporte", f"{rule['support']:.1%}", 
                                     help="Frequência desta combinação")
                        with col2:
                            st.metric("Confiança", f"{rule['confidence']:.1%}",
                                     help="Probabilidade do consequente dado o antecedente")
                        with col3:
                            st.metric("Lift", f"{rule['lift']:.2f}",
                                     help=">1 significa correlação positiva")
                        
                        st.markdown(f"""
                        **Interpretação:**
                        - Quando ocorrem: **{antecedents}**
                        - Há {rule['confidence']*100:.0f}% de chance de também ocorrer: **{consequents}**
                        - Esta combinação é {rule['lift']:.1f}x mais provável do que o acaso
                        """)
                        
                        if rule['lift'] > 2.0:
                            st.error("🚨 **REGRA CRÍTICA**: Lift muito alto indica forte associação de risco!")
                
                # Tabela completa
                with st.expander("📋 Todas as Regras (Tabela Completa)", expanded=False):
                    rules_display = association_results['rules'].copy()
                    rules_display['antecedents'] = rules_display['antecedents'].apply(
                        lambda x: ', '.join([item.replace('_', ' ') for item in list(x)])
                    )
                    rules_display['consequents'] = rules_display['consequents'].apply(
                        lambda x: ', '.join([item.replace('_', ' ') for item in list(x)])
                    )
                    
                    st.dataframe(
                        rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
                        use_container_width=True
                    )
                
        except Exception as e:
            st.warning(f"Regras de associação indisponíveis: {str(e)}")
    
    # ---- SUBTAB 5: PADRÕES COMBINADOS ----
    with subtab5:
        st.subheader("📈 Padrões Combinados (Análise Clássica)")
        
        st.markdown("""
        **Análise de padrões usando agregações estatísticas** para complementar os algoritmos avançados.
        """)
        
        try:
            with st.spinner("Minerando padrões de associação..."):
                patterns = unsupervised.find_risk_patterns(df_filtered)
            
            if patterns['total_patterns_found'] == 0:
                st.info("Nenhum padrão significativo encontrado com os filtros atuais.")
            else:
                st.success(f"✅ {patterns['total_patterns_found']} tipo(s) de padrão descoberto(s)!")
                
                # Exibir cada padrão
                for idx, pattern in enumerate(patterns['patterns']):
                    st.markdown(f"### {pattern['type']}")
                    st.markdown(f"*{pattern['description']}*")
                    fig_pattern = create_pattern_network(patterns)
                    st.plotly_chart(fig_pattern, use_container_width=True, key=f"pattern_{idx}")
                    with st.expander("📋 Ver Dados Detalhados"):
                        st.dataframe(pattern['data'], use_container_width=True)
                    st.markdown("---")
                
                # Insights práticos
                st.subheader("💡 Insights para Operações")
                
                st.markdown("""
                **Como usar esses padrões:**
                
                1. **Priorização de Patrulhamento:** Focar nos locais/horários dos padrões de alto risco
                2. **Alocação de Recursos:** Quando detectar padrão crítico, enviar backup preventivo
                3. **Treinamento:** Preparar equipes para situações típicas de cada padrão
                4. **Prevenção:** Identificar padrões emergentes antes que se tornem tendência
                """)
                
        except Exception as e:
            st.error(f"Erro na análise de padrões: {str(e)}")
    
    # Resumo geral
    st.markdown("---")
    st.info("""
    **🎓 Diferença entre Análise Supervisionada e Não Supervisionada:**
    
    - **Supervisionada (Tabs anteriores):** "Aprende" o que você definiu como risco (arma de fogo/explosivos) e prevê ocorrências futuras
    - **Não Supervisionada (Esta tab):** "Descobre" padrões escondidos que você talvez não sabia que existiam
    
    **Complementaridade:** Use ambas! A supervisionada prevê, a não supervisionada descobre novos riscos.
    """)

# Footer
st.markdown("---")
st.markdown("**🚓 Delegacia 5.0** - Sistema de Inteligência para Segurança Policial")
st.markdown("*Protegendo quem nos protege através de dados e IA*")
