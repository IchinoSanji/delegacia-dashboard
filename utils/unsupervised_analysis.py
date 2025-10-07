"""
Módulo de análise não supervisionada para descoberta de padrões de risco.
Implementa clustering, detecção de anomalias e análise de associações.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedRiskAnalyzer:
    """Analisador não supervisionado de padrões de risco operacional."""
    
    def __init__(self, random_state=42):
        """
        Inicializa o analisador não supervisionado.
        
        Args:
            random_state (int): Semente para reprodutibilidade
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.tsne = None
        self.kmeans = None
        self.dbscan = None
        self.isolation_forest = None
        self.feature_names = None
        
    def prepare_features(self, df):
        """
        Prepara features numéricas para análise não supervisionada.
        
        Args:
            df (pandas.DataFrame): Dados de entrada
            
        Returns:
            numpy.ndarray: Features preparadas
        """
        # Features numéricas
        numeric_features = []
        feature_names = []
        
        # Adicionar features numéricas diretas
        num_cols = ['quantidade_vitimas', 'quantidade_suspeitos', 'idade_suspeito', 
                   'hora', 'dia_semana_idx', 'risk_score']
        
        for col in num_cols:
            if col in df.columns:
                numeric_features.append(df[col].fillna(df[col].median()).values)
                feature_names.append(col)
        
        # Encoding de features categóricas importantes
        if 'arma_utilizada' in df.columns:
            weapon_encoded = (df['arma_utilizada'].isin(['Arma de Fogo', 'Explosivos'])).astype(int).values
            numeric_features.append(weapon_encoded)
            feature_names.append('arma_letal')
        
        if 'tipo_crime' in df.columns:
            violent_crimes = ['Homicídio', 'Latrocínio', 'Roubo', 'Sequestro', 'Tráfico de Drogas']
            crime_encoded = df['tipo_crime'].isin(violent_crimes).astype(int).values
            numeric_features.append(crime_encoded)
            feature_names.append('crime_violento')
        
        if 'turno' in df.columns:
            night_shift = df['turno'].isin(['Noite', 'Madrugada']).astype(int).values
            numeric_features.append(night_shift)
            feature_names.append('periodo_noturno')
        
        # Combinar features
        if not numeric_features:
            raise ValueError("Nenhuma feature numérica disponível")
        
        X = np.column_stack(numeric_features)
        self.feature_names = feature_names
        
        return X
    
    def find_optimal_clusters(self, df, max_clusters=10):
        """
        Determina o número ótimo de clusters usando método do cotovelo e silhouette.
        
        Args:
            df (pandas.DataFrame): Dados de entrada
            max_clusters (int): Número máximo de clusters a testar
            
        Returns:
            dict: Métricas para cada número de clusters
        """
        try:
            X = self.prepare_features(df)
            X_scaled = self.scaler.fit_transform(X)
            
            # Garantir que temos dados suficientes
            n_samples = len(df)
            if n_samples < 4:
                raise ValueError("Dados insuficientes para análise do cotovelo (mínimo: 4 amostras)")
            
            # Calcular range válido de clusters
            min_k = 2
            max_k = min(max_clusters + 1, max(3, n_samples // 5))  # Pelo menos 3, ou n/5
            
            if max_k < min_k:
                raise ValueError(f"Dataset muito pequeno ({n_samples} amostras). Necessário pelo menos 10 amostras para análise.")
            
            k_range = range(min_k, max_k)
            
            inertias = []
            silhouettes = []
            davies_bouldin = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
                labels = kmeans.fit_predict(X_scaled)
                
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(X_scaled, labels))
                davies_bouldin.append(davies_bouldin_score(X_scaled, labels))
            
            # Encontrar cotovelo (método da segunda derivada)
            if len(inertias) >= 3:
                diffs = np.diff(inertias)
                second_diffs = np.diff(diffs)
                elbow_idx = np.argmax(second_diffs) + 2
                optimal_k = list(k_range)[min(elbow_idx, len(k_range) - 1)]
            else:
                optimal_k = list(k_range)[np.argmax(silhouettes)]
            
            results = {
                'k_range': list(k_range),
                'inertias': inertias,
                'silhouettes': silhouettes,
                'davies_bouldin': davies_bouldin,
                'optimal_k': optimal_k,
                'best_silhouette_k': list(k_range)[np.argmax(silhouettes)]
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Erro na busca de clusters ótimos: {str(e)}")
    
    def perform_clustering(self, df, n_clusters=4, method='kmeans'):
        """
        Realiza clustering para descobrir grupos de risco.
        
        Args:
            df (pandas.DataFrame): Dados de entrada
            n_clusters (int): Número de clusters (para K-Means)
            method (str): 'kmeans' ou 'dbscan'
            
        Returns:
            dict: Resultados do clustering
        """
        try:
            # Preparar features
            X = self.prepare_features(df)
            X_scaled = self.scaler.fit_transform(X)
            
            if method == 'kmeans':
                # K-Means clustering
                self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init='auto')
                clusters = self.kmeans.fit_predict(X_scaled)
                method_name = 'K-Means'
                
            elif method == 'dbscan':
                # DBSCAN clustering
                self.dbscan = DBSCAN(eps=0.5, min_samples=5)
                clusters = self.dbscan.fit_predict(X_scaled)
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                method_name = 'DBSCAN'
            else:
                raise ValueError(f"Método desconhecido: {method}")
            
            # Calcular silhouette score
            if len(np.unique(clusters)) > 1 and -1 not in clusters:
                silhouette = silhouette_score(X_scaled, clusters)
            else:
                silhouette = 0
            
            # PCA para visualização
            self.pca = PCA(n_components=2, random_state=self.random_state)
            X_pca = self.pca.fit_transform(X_scaled)
            
            # Analisar características de cada cluster
            df_clustered = df.copy()
            df_clustered['cluster'] = clusters
            df_clustered['pca_1'] = X_pca[:, 0]
            df_clustered['pca_2'] = X_pca[:, 1]
            
            # Estatísticas por cluster
            cluster_stats = []
            unique_clusters = sorted([c for c in np.unique(clusters) if c != -1])
            
            for cluster_id in unique_clusters:
                cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                
                stats = {
                    'cluster_id': cluster_id,
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(df) * 100,
                    'avg_risk_score': cluster_data['risk_score'].mean() if 'risk_score' in cluster_data.columns else 0,
                    'high_risk_rate': cluster_data['risco_alto'].mean() if 'risco_alto' in cluster_data.columns else 0,
                    'avg_suspects': cluster_data['quantidade_suspeitos'].mean() if 'quantidade_suspeitos' in cluster_data.columns else 0,
                    'weapon_rate': (cluster_data['arma_utilizada'].isin(['Arma de Fogo', 'Explosivos'])).mean() if 'arma_utilizada' in cluster_data.columns else 0,
                    'most_common_crime': cluster_data['tipo_crime'].mode()[0] if 'tipo_crime' in cluster_data.columns and len(cluster_data) > 0 else 'N/A',
                    'most_common_bairro': cluster_data['bairro'].mode()[0] if 'bairro' in cluster_data.columns and len(cluster_data) > 0 else 'N/A',
                    'most_common_shift': cluster_data['turno'].mode()[0] if 'turno' in cluster_data.columns and len(cluster_data) > 0 else 'N/A'
                }
                cluster_stats.append(stats)
            
            results = {
                'df_clustered': df_clustered,
                'cluster_stats': pd.DataFrame(cluster_stats),
                'silhouette_score': silhouette,
                'n_clusters': n_clusters,
                'pca_variance_explained': self.pca.explained_variance_ratio_.sum(),
                'method': method_name
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Erro no clustering: {str(e)}")
    
    def perform_tsne(self, df, perplexity=30):
        """
        Realiza redução dimensional com t-SNE para visualização alternativa.
        
        Args:
            df (pandas.DataFrame): Dados de entrada
            perplexity (int): Parâmetro de perplexidade do t-SNE
            
        Returns:
            dict: Resultados do t-SNE
        """
        try:
            X = self.prepare_features(df)
            X_scaled = self.scaler.fit_transform(X)
            
            n_samples = len(df)
            
            # Verificar se há amostras suficientes
            if n_samples < 3:
                raise ValueError("t-SNE requer pelo menos 3 amostras")
            
            # Ajustar perplexity para o intervalo válido: [5, n_samples-1]
            # Perplexity deve ser menor que n_samples
            min_perplexity = 5
            max_allowed_perplexity = max(1, n_samples - 1)  # t-SNE requer perplexity < n_samples
            
            # Se dataset muito pequeno, usar valor menor
            if max_allowed_perplexity < min_perplexity:
                actual_perplexity = max(1, min(perplexity, max_allowed_perplexity))
            else:
                actual_perplexity = max(min_perplexity, min(perplexity, max_allowed_perplexity))
            
            self.tsne = TSNE(n_components=2, random_state=self.random_state, 
                 perplexity=actual_perplexity)
            X_tsne = self.tsne.fit_transform(X_scaled)
            
            df_tsne = df.copy()
            df_tsne['tsne_1'] = X_tsne[:, 0]
            df_tsne['tsne_2'] = X_tsne[:, 1]
            
            return {
                'df_tsne': df_tsne,
                'perplexity_used': actual_perplexity
            }
            
        except Exception as e:
            raise Exception(f"Erro no t-SNE: {str(e)}")
    
    def perform_pca_analysis(self, df, n_components=None):
        """
        Análise completa de PCA com scree plot e variância explicada.
        
        Args:
            df (pandas.DataFrame): Dados de entrada
            n_components (int): Número de componentes (None = todos)
            
        Returns:
            dict: Análise completa de PCA
        """
        try:
            X = self.prepare_features(df)
            X_scaled = self.scaler.fit_transform(X)
            
            # PCA completo
            n_comp = min(X_scaled.shape[1], X_scaled.shape[0]) if n_components is None else n_components
            pca_full = PCA(n_components=n_comp, random_state=self.random_state)
            X_pca = pca_full.fit_transform(X_scaled)
            
            # Calcular variância explicada acumulada
            explained_var = pca_full.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            # Número de componentes para 90% de variância
            n_comp_90 = np.argmax(cumulative_var >= 0.90) + 1
            
            # Loadings (contribuição de cada feature)
            loadings = pd.DataFrame(
                pca_full.components_.T,
                columns=[f'PC{i+1}' for i in range(n_comp)],
                index=self.feature_names
            )
            
            results = {
                'explained_variance': explained_var,
                'cumulative_variance': cumulative_var,
                'n_components_90': n_comp_90,
                'loadings': loadings,
                'transformed_data': X_pca,
                'n_components': n_comp
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Erro na análise PCA: {str(e)}")
    
    def detect_anomalies(self, df, contamination=0.1):
        """
        Detecta anomalias usando Isolation Forest.
        
        Args:
            df (pandas.DataFrame): Dados de entrada
            contamination (float): Proporção esperada de anomalias
            
        Returns:
            dict: Resultados da detecção de anomalias
        """
        try:
            # Preparar features
            X = self.prepare_features(df)
            
            # Normalizar
            X_scaled = self.scaler.fit_transform(X)
            
            # Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=self.random_state,
                n_estimators=100
            )
            
            # -1 para anomalias, 1 para normais
            predictions = self.isolation_forest.fit_predict(X_scaled)
            anomaly_scores = self.isolation_forest.score_samples(X_scaled)
            
            # Adicionar ao dataframe
            df_anomalies = df.copy()
            df_anomalies['is_anomaly'] = (predictions == -1).astype(int)
            df_anomalies['anomaly_score'] = -anomaly_scores  # Inverter para que maior = mais anômalo
            
            # Estatísticas
            anomalies = df_anomalies[df_anomalies['is_anomaly'] == 1]
            normal = df_anomalies[df_anomalies['is_anomaly'] == 0]
            
            stats = {
                'total_anomalies': len(anomalies),
                'anomaly_percentage': len(anomalies) / len(df) * 100,
                'anomalies_high_risk_rate': anomalies['risco_alto'].mean() if 'risco_alto' in anomalies.columns and len(anomalies) > 0 else 0,
                'normal_high_risk_rate': normal['risco_alto'].mean() if 'risco_alto' in normal.columns and len(normal) > 0 else 0,
                'top_anomaly_crimes': anomalies['tipo_crime'].value_counts().head(5).to_dict() if 'tipo_crime' in anomalies.columns and len(anomalies) > 0 else {},
                'top_anomaly_locations': anomalies['bairro'].value_counts().head(5).to_dict() if 'bairro' in anomalies.columns and len(anomalies) > 0 else {}
            }
            
            # Top anomalias mais severas
            top_anomalies = df_anomalies.nlargest(20, 'anomaly_score')
            
            results = {
                'df_anomalies': df_anomalies,
                'stats': stats,
                'top_anomalies': top_anomalies,
                'contamination': contamination
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Erro na detecção de anomalias: {str(e)}")
    
    def find_association_rules(self, df, min_support=0.05, min_confidence=0.5):
        """
        Encontra regras de associação usando algoritmo Apriori.
        
        Args:
            df (pandas.DataFrame): Dados de entrada
            min_support (float): Suporte mínimo
            min_confidence (float): Confiança mínima
            
        Returns:
            dict: Regras de associação encontradas
        """
        try:
            # Preparar transações (apenas ocorrências de alto risco)
            df_high_risk = df[df['risco_alto'] == 1].copy() if 'risco_alto' in df.columns else df.copy()
            
            if len(df_high_risk) < 10:
                return {'rules': pd.DataFrame(), 'total_rules': 0}
            
            # Criar transações com características categóricas
            transactions = []
            for _, row in df_high_risk.iterrows():
                transaction = []
                
                if 'bairro' in row and pd.notna(row['bairro']):
                    transaction.append(f"Bairro_{row['bairro']}")
                if 'tipo_crime' in row and pd.notna(row['tipo_crime']):
                    transaction.append(f"Crime_{row['tipo_crime']}")
                if 'arma_utilizada' in row and pd.notna(row['arma_utilizada']):
                    transaction.append(f"Arma_{row['arma_utilizada']}")
                if 'turno' in row and pd.notna(row['turno']):
                    transaction.append(f"Turno_{row['turno']}")
                if 'dia_semana' in row and pd.notna(row['dia_semana']):
                    transaction.append(f"Dia_{row['dia_semana']}")
                
                if transaction:
                    transactions.append(transaction)
            
            if not transactions or len(transactions) < 5:
                return {'rules': pd.DataFrame(), 'total_rules': 0}
            
            # Codificar transações
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Encontrar itemsets frequentes
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) == 0:
                return {'rules': pd.DataFrame(), 'total_rules': 0}
            
            # Gerar regras de associação
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            if len(rules) > 0:
                # Adicionar métricas de interesse
                rules['lift'] = rules['lift'].round(2)
                rules['confidence'] = rules['confidence'].round(2)
                rules['support'] = rules['support'].round(3)
                
                # Ordenar por lift
                rules = rules.sort_values('lift', ascending=False)
            
            results = {
                'rules': rules,
                'total_rules': len(rules),
                'frequent_itemsets': frequent_itemsets
            }
            
            return results
            
        except Exception as e:
            return {'rules': pd.DataFrame(), 'total_rules': 0, 'error': str(e)}
    
    def find_risk_patterns(self, df):
        """
        Encontra padrões de associação entre variáveis de risco.
        
        Args:
            df (pandas.DataFrame): Dados de entrada
            
        Returns:
            dict: Padrões descobertos
        """
        try:
            patterns = []
            
            # Padrão 1: Bairro + Turno + Taxa de Risco
            if all(col in df.columns for col in ['bairro', 'turno', 'risco_alto']):
                bairro_turno = df.groupby(['bairro', 'turno']).agg({
                    'risco_alto': ['count', 'sum', 'mean']
                }).reset_index()
                bairro_turno.columns = ['bairro', 'turno', 'total', 'alto_risco', 'taxa_risco']
                
                # Filtrar padrões significativos (mais de 10 casos e taxa > 30%)
                significant = bairro_turno[(bairro_turno['total'] >= 10) & (bairro_turno['taxa_risco'] >= 0.3)]
                
                if len(significant) > 0:
                    patterns.append({
                        'type': 'Bairro + Turno de Alto Risco',
                        'data': significant.sort_values('taxa_risco', ascending=False).head(10),
                        'description': 'Combinações de local e horário com alta taxa de risco'
                    })
            
            # Padrão 2: Tipo de Crime + Arma + Bairro
            if all(col in df.columns for col in ['tipo_crime', 'arma_utilizada', 'bairro']):
                crime_weapon = df.groupby(['tipo_crime', 'arma_utilizada', 'bairro']).size().reset_index(name='count')
                
                # Top combinações
                top_combinations = crime_weapon.nlargest(15, 'count')
                
                if len(top_combinations) > 0:
                    patterns.append({
                        'type': 'Crime + Arma + Local Frequente',
                        'data': top_combinations,
                        'description': 'Combinações mais frequentes de crime, arma e local'
                    })
            
            # Padrão 3: Dia da Semana + Hora + Tipo Crime
            if all(col in df.columns for col in ['dia_semana', 'hora', 'tipo_crime', 'risco_alto']):
                temporal_crime = df.groupby(['dia_semana', 'hora', 'tipo_crime']).agg({
                    'risco_alto': ['count', 'mean']
                }).reset_index()
                temporal_crime.columns = ['dia_semana', 'hora', 'tipo_crime', 'count', 'taxa_risco']
                
                # Filtrar padrões de alto risco
                high_risk_patterns = temporal_crime[(temporal_crime['count'] >= 5) & (temporal_crime['taxa_risco'] >= 0.5)]
                
                if len(high_risk_patterns) > 0:
                    patterns.append({
                        'type': 'Padrões Temporais de Alto Risco',
                        'data': high_risk_patterns.sort_values('taxa_risco', ascending=False).head(10),
                        'description': 'Combinações de dia, hora e crime com alta taxa de risco'
                    })
            
            return {
                'patterns': patterns,
                'total_patterns_found': len(patterns)
            }
            
        except Exception as e:
            return {'patterns': [], 'total_patterns_found': 0, 'error': str(e)}
    
    def get_cluster_interpretation(self, cluster_stats):
        """
        Interpreta os clusters encontrados.
        
        Args:
            cluster_stats (pandas.DataFrame): Estatísticas dos clusters
            
        Returns:
            list: Interpretações dos clusters
        """
        interpretations = []
        
        for _, cluster in cluster_stats.iterrows():
            cluster_id = cluster['cluster_id']
            risk_rate = cluster['high_risk_rate'] * 100
            weapon_rate = cluster['weapon_rate'] * 100
            
            # Determinar perfil do cluster
            if risk_rate >= 50 and weapon_rate >= 40:
                profile = "🔴 CLUSTER CRÍTICO"
                description = f"Alto risco operacional - {risk_rate:.1f}% com armas letais em {weapon_rate:.1f}% dos casos"
                recommendation = "Requer protocolo de máxima segurança e backup obrigatório"
            elif risk_rate >= 30:
                profile = "🟡 CLUSTER DE ATENÇÃO"
                description = f"Risco elevado - {risk_rate:.1f}% de situações perigosas"
                recommendation = "Manter vigilância e considerar backup preventivo"
            else:
                profile = "🟢 CLUSTER PADRÃO"
                description = f"Risco moderado/baixo - {risk_rate:.1f}% de alto risco"
                recommendation = "Abordagem padrão com cautela normal"
            
            interpretations.append({
                'cluster_id': cluster_id,
                'profile': profile,
                'description': description,
                'recommendation': recommendation,
                'characteristics': {
                    'crime_comum': cluster['most_common_crime'],
                    'bairro_comum': cluster['most_common_bairro'],
                    'turno_comum': cluster['most_common_shift'],
                    'tamanho': f"{cluster['percentage']:.1f}% dos casos"
                }
            })
        
        return interpretations
