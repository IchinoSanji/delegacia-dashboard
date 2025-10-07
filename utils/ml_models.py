"""
Modelos de Machine Learning para predição de risco operacional policial.
Foca na identificação de situações de alto risco para oficiais.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class RiskPredictor:
    """Modelo de predição de risco operacional para policiais."""
    
    def __init__(self, random_state=42):
        """
        Inicializa o preditor de risco.
        
        Args:
            random_state (int): Semente para reprodutibilidade
        """
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.is_trained = False
        
        # Configurações padrão
        self.numeric_features = [
            'quantidade_vitimas', 'quantidade_suspeitos', 
            'idade_suspeito', 'hora', 'dia_semana_idx'
        ]
        
        self.categorical_features = [
            'bairro', 'tipo_crime', 'sexo_suspeito', 
            'turno', 'orgao_responsavel', 'status_investigacao'
        ]
    
    def _prepare_features(self, df):
        """
        Prepara as features para o modelo.
        
        Args:
            df (pandas.DataFrame): Dados de entrada
            
        Returns:
            tuple: (X, y) features e target
        """
        # Verificar features disponíveis
        available_numeric = [f for f in self.numeric_features if f in df.columns]
        available_categorical = [f for f in self.categorical_features if f in df.columns]
        
        all_features = available_numeric + available_categorical
        
        if not all_features:
            raise ValueError("Nenhuma feature válida encontrada nos dados")
        
        # Preparar X e y
        X = df[all_features].copy()
        y = df['risco_alto'].copy() if 'risco_alto' in df.columns else None
        
        # Armazenar nomes das features para referência
        self.feature_names = all_features
        
        return X, y, available_numeric, available_categorical
    
    def _create_preprocessor(self, numeric_features, categorical_features):
        """
        Cria o preprocessador de dados.
        
        Args:
            numeric_features (list): Lista de features numéricas
            categorical_features (list): Lista de features categóricas
            
        Returns:
            ColumnTransformer: Preprocessador configurado
        """
        transformers = []
        
        if numeric_features:
            transformers.append(
                ('num', StandardScaler(), numeric_features)
            )
        
        if categorical_features:
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            )
        
        if not transformers:
            raise ValueError("Não foi possível criar preprocessador - nenhuma feature válida")
        
        return ColumnTransformer(transformers=transformers, remainder='drop')
    
    def train_model(self, df, test_size=0.2):
        """
        Treina o modelo de predição de risco.
        
        Args:
            df (pandas.DataFrame): Dados de treino
            test_size (float): Proporção dos dados para teste
            
        Returns:
            dict: Métricas de performance
        """
        try:
            # Verificar se há target
            if 'risco_alto' not in df.columns:
                raise ValueError("Coluna 'risco_alto' não encontrada nos dados")
            
            # Preparar dados
            X, y, numeric_features, categorical_features = self._prepare_features(df)
            
            if len(X) < 50:
                raise ValueError("Dados insuficientes para treinar modelo")
            
            # Split temporal se possível, senão aleatório
            if 'data_ocorrencia' in df.columns:
                # Split temporal (70% treino, 30% teste)
                cutoff_date = df['data_ocorrencia'].quantile(0.7)
                train_mask = df['data_ocorrencia'] <= cutoff_date
                
                X_train, X_test = X[train_mask], X[~train_mask]
                y_train, y_test = y[train_mask], y[~train_mask]
                
                # Se split muito desbalanceado, usar split aleatório
                if len(X_train) < 30 or len(X_test) < 10:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=self.random_state,
                        stratify=y if y.nunique() > 1 else None
                    )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state,
                    stratify=y if y.nunique() > 1 else None
                )
            
            # Criar preprocessador
            self.preprocessor = self._create_preprocessor(numeric_features, categorical_features)
            
            # Criar e treinar modelo
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Pipeline completo
            self.model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            # Treinar modelo
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Avaliar performance
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_test.nunique() > 1 else 0,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            return metrics
            
        except Exception as e:
            raise Exception(f"Erro ao treinar modelo: {str(e)}")
    
    def predict_risk(self, df):
        """
        Prediz o risco para novos dados.
        
        Args:
            df (pandas.DataFrame): Dados para predição
            
        Returns:
            dict: Predições e probabilidades
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda")
        
        try:
            X, _, _, _ = self._prepare_features(df)
            
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'high_risk_count': int(predictions.sum()),
                'avg_risk_score': float(probabilities.mean())
            }
            
        except Exception as e:
            raise Exception(f"Erro na predição: {str(e)}")
    
    def get_feature_importance(self):
        """
        Obtém a importância das features.
        
        Returns:
            pandas.DataFrame: Features e suas importâncias
        """
        if not self.is_trained:
            return pd.DataFrame()
        
        try:
            # Obter importâncias do RandomForest
            rf_model = self.model.named_steps['classifier']
            
            # Obter nomes das features após preprocessing
            feature_names = self._get_feature_names_after_preprocessing()
            
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                
                # Criar DataFrame com importâncias
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                return importance_df
            else:
                return pd.DataFrame()
                
        except Exception:
            return pd.DataFrame()
    
    def _get_feature_names_after_preprocessing(self):
        """
        Obtém os nomes das features após o preprocessing.
        
        Returns:
            list: Lista de nomes das features
        """
        try:
            if self.preprocessor is None:
                return []
            
            feature_names = []
            
            # Features numéricas
            if 'num' in dict(self.preprocessor.transformers):
                numeric_transformer = dict(self.preprocessor.transformers)['num']
                if hasattr(numeric_transformer, 'get_feature_names_out'):
                    numeric_names = numeric_transformer.get_feature_names_out()
                else:
                    # Fallback para versões antigas do sklearn
                    numeric_names = [f for f in self.feature_names if f in self.numeric_features]
                feature_names.extend(numeric_names)
            
            # Features categóricas (one-hot encoded)
            if 'cat' in dict(self.preprocessor.transformers):
                cat_transformer = dict(self.preprocessor.transformers)['cat']
                if hasattr(cat_transformer, 'get_feature_names_out'):
                    cat_names = cat_transformer.get_feature_names_out()
                else:
                    # Fallback - criar nomes aproximados
                    cat_features = [f for f in self.feature_names if f in self.categorical_features]
                    cat_names = [f"cat__{f}_encoded" for f in cat_features]
                feature_names.extend(cat_names)
            
            return feature_names
            
        except Exception:
            # Fallback - usar nomes originais
            return self.feature_names if self.feature_names else []
    
    def calculate_f1_at_k(self, df, k=0.20):
        """
        Calcula F1-Score priorizando top K% das predições.
        
        Args:
            df (pandas.DataFrame): Dados para avaliação
            k (float): Proporção de dados a priorizar (0-1)
            
        Returns:
            dict: Métricas F1@K
        """
        if not self.is_trained:
            return {'f1_score': 0, 'precision': 0, 'recall': 0}
        
        try:
            # Predizer probabilidades
            pred_results = self.predict_risk(df)
            probabilities = pred_results['probabilities']
            
            # Obter target real
            if 'risco_alto' not in df.columns:
                return {'f1_score': 0, 'precision': 0, 'recall': 0}
            
            y_true = df['risco_alto'].values
            
            # Calcular threshold para top k%
            k_threshold = np.percentile(probabilities, (1 - k) * 100)
            
            # Criar predições binárias baseadas no threshold
            y_pred_k = (probabilities >= k_threshold).astype(int)
            
            # Calcular métricas
            precision = precision_score(y_true, y_pred_k, zero_division=0)
            recall = recall_score(y_true, y_pred_k, zero_division=0)
            f1 = f1_score(y_true, y_pred_k, zero_division=0)
            
            return {
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'threshold': k_threshold,
                'selected_count': int(y_pred_k.sum()),
                'total_high_risk': int(y_true.sum())
            }
            
        except Exception as e:
            return {'f1_score': 0, 'precision': 0, 'recall': 0, 'error': str(e)}
    
    def get_performance_curve(self, df, k_values=None):
        """
        Gera curva de performance para diferentes valores de K.
        
        Args:
            df (pandas.DataFrame): Dados para avaliação
            k_values (list): Lista de valores K para testar
            
        Returns:
            dict: Dados da curva de performance
        """
        if k_values is None:
            k_values = np.linspace(0.1, 0.5, 21)  # 10% a 50%
        
        results = {
            'k_values': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        for k in k_values:
            metrics = self.calculate_f1_at_k(df, k)
            
            results['k_values'].append(k * 100)  # Converter para porcentagem
            results['precision'].append(metrics['precision'])
            results['recall'].append(metrics['recall'])
            results['f1_score'].append(metrics['f1_score'])
        
        return results
    
    def get_model_summary(self):
        """
        Retorna resumo do modelo treinado.
        
        Returns:
            dict: Resumo do modelo
        """
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        try:
            rf_model = self.model.named_steps['classifier']
            
            summary = {
                'status': 'trained',
                'model_type': 'RandomForest',
                'n_estimators': rf_model.n_estimators,
                'max_depth': rf_model.max_depth,
                'features_used': len(self.feature_names) if self.feature_names else 0,
                'feature_names': self.feature_names[:10] if self.feature_names else [],  # Top 10
                'class_weight': str(rf_model.class_weight)
            }
            
            return summary
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
