"""
Processador de dados para o sistema de risco operacional policial.
Foca no pré-processamento e limpeza dos dados de ocorrências.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Processador de dados para análise de risco operacional policial."""
    
    def __init__(self, csv_path):
        """
        Inicializa o processador com o caminho do CSV.
        
        Args:
            csv_path (str): Caminho para o arquivo CSV
        """
        self.csv_path = csv_path
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Carrega e processa os dados iniciais."""
        try:
            self.df = pd.read_csv(self.csv_path)
            self._initial_processing()
        except Exception as e:
            raise Exception(f"Erro ao carregar dados: {str(e)}")
    
    def _initial_processing(self):
        """Processamento inicial dos dados."""
        if self.df is None:
            return
        
        # Converter data_ocorrencia para datetime
        if 'data_ocorrencia' in self.df.columns:
            self.df['data_ocorrencia'] = pd.to_datetime(self.df['data_ocorrencia'], errors='coerce')
        
        # Criar variáveis temporais
        if 'data_ocorrencia' in self.df.columns:
            self.df['ano'] = self.df['data_ocorrencia'].dt.year
            self.df['mes'] = self.df['data_ocorrencia'].dt.month
            self.df['dia'] = self.df['data_ocorrencia'].dt.day
            self.df['hora'] = self.df['data_ocorrencia'].dt.hour
            self.df['dia_semana'] = self.df['data_ocorrencia'].dt.day_name()
            self.df['dia_semana_idx'] = self.df['data_ocorrencia'].dt.dayofweek
            
            # Criar variável de turno
            self.df['turno'] = self.df['hora'].apply(self._get_turno)
        
        # Criar variável de risco alto (target)
        if 'arma_utilizada' in self.df.columns:
            self.df['risco_alto'] = (
                self.df['arma_utilizada'].isin(['Arma de Fogo', 'Explosivos'])
            ).astype(int)
        
        # Limpar dados categóricos
        categorical_cols = [
            'bairro', 'tipo_crime', 'sexo_suspeito', 
            'orgao_responsavel', 'status_investigacao', 'descricao_modus_operandi'
        ]
        
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Não Informado')
        
        # Limpar dados numéricos
        numeric_cols = [
            'quantidade_vitimas', 'quantidade_suspeitos', 
            'idade_suspeito', 'latitude', 'longitude'
        ]
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df[col] = self.df[col].fillna(self.df[col].median())
    
    def _get_turno(self, hora):
        """
        Converte hora em turno de trabalho.
        
        Args:
            hora (int): Hora do dia (0-23)
            
        Returns:
            str: Nome do turno
        """
        if pd.isna(hora):
            return 'Não Informado'
        
        if 6 <= hora < 12:
            return 'Manhã'
        elif 12 <= hora < 18:
            return 'Tarde'
        elif 18 <= hora < 24:
            return 'Noite'
        else:  # 0 <= hora < 6
            return 'Madrugada'
    
    def get_processed_data(self):
        """
        Retorna os dados processados.
        
        Returns:
            pandas.DataFrame: Dados processados
        """
        return self.df.copy() if self.df is not None else None
    
    def get_risk_statistics(self):
        """
        Calcula estatísticas de risco.
        
        Returns:
            dict: Dicionário com estatísticas
        """
        if self.df is None or 'risco_alto' not in self.df.columns:
            return {}
        
        stats = {
            'total_ocorrencias': len(self.df),
            'total_alto_risco': self.df['risco_alto'].sum(),
            'taxa_risco_geral': self.df['risco_alto'].mean(),
            'bairros_unicos': self.df['bairro'].nunique() if 'bairro' in self.df.columns else 0,
            'periodo_analise': {
                'inicio': self.df['data_ocorrencia'].min() if 'data_ocorrencia' in self.df.columns else None,
                'fim': self.df['data_ocorrencia'].max() if 'data_ocorrencia' in self.df.columns else None
            }
        }
        
        return stats
    
    def get_bairro_risk_summary(self):
        """
        Gera resumo de risco por bairro.
        
        Returns:
            pandas.DataFrame: Resumo por bairro
        """
        if self.df is None or 'bairro' not in self.df.columns:
            return pd.DataFrame()
        
        summary = self.df.groupby('bairro').agg({
            'risco_alto': ['count', 'sum', 'mean'],
            'quantidade_suspeitos': 'mean',
            'quantidade_vitimas': 'mean'
        }).round(3)
        
        summary.columns = [
            'Total_Ocorrencias', 'Alto_Risco', 'Taxa_Risco',
            'Media_Suspeitos', 'Media_Vitimas'
        ]
        
        summary = summary.sort_values('Taxa_Risco', ascending=False)
        return summary.reset_index()
    
    def get_temporal_patterns(self):
        """
        Analisa padrões temporais de risco.
        
        Returns:
            dict: Padrões temporais
        """
        if self.df is None:
            return {}
        
        patterns = {}
        
        # Por turno
        if 'turno' in self.df.columns:
            patterns['por_turno'] = self.df.groupby('turno').agg({
                'risco_alto': ['count', 'sum', 'mean']
            }).round(3)
        
        # Por dia da semana
        if 'dia_semana' in self.df.columns:
            patterns['por_dia_semana'] = self.df.groupby('dia_semana').agg({
                'risco_alto': ['count', 'sum', 'mean']
            }).round(3)
        
        # Por hora
        if 'hora' in self.df.columns:
            patterns['por_hora'] = self.df.groupby('hora').agg({
                'risco_alto': ['count', 'sum', 'mean']
            }).round(3)
        
        return patterns
    
    def filter_data(self, filters):
        """
        Aplica filtros aos dados.
        
        Args:
            filters (dict): Dicionário de filtros
            
        Returns:
            pandas.DataFrame: Dados filtrados
        """
        if self.df is None:
            return pd.DataFrame()
        
        filtered_df = self.df.copy()
        
        # Filtro por data
        if 'date_range' in filters and filters['date_range']:
            start_date, end_date = filters['date_range']
            filtered_df = filtered_df[
                (filtered_df['data_ocorrencia'].dt.date >= start_date) &
                (filtered_df['data_ocorrencia'].dt.date <= end_date)
            ]
        
        # Filtro por bairro
        if 'bairro' in filters and filters['bairro'] != 'Todos':
            filtered_df = filtered_df[filtered_df['bairro'] == filters['bairro']]
        
        # Filtro por tipo de crime
        if 'tipo_crime' in filters and filters['tipo_crime'] != 'Todos':
            filtered_df = filtered_df[filtered_df['tipo_crime'] == filters['tipo_crime']]
        
        # Filtro por turno
        if 'turnos' in filters and filters['turnos']:
            filtered_df = filtered_df[filtered_df['turno'].isin(filters['turnos'])]
        
        return filtered_df
    
    def validate_data_quality(self):
        """
        Valida a qualidade dos dados.
        
        Returns:
            dict: Relatório de qualidade
        """
        if self.df is None:
            return {'status': 'error', 'message': 'Dados não carregados'}
        
        quality_report = {
            'status': 'ok',
            'total_rows': len(self.df),
            'missing_data': {},
            'data_types': {},
            'warnings': []
        }
        
        # Verificar dados ausentes
        for col in self.df.columns:
            missing_pct = (self.df[col].isna().sum() / len(self.df)) * 100
            if missing_pct > 0:
                quality_report['missing_data'][col] = f"{missing_pct:.1f}%"
        
        # Verificar tipos de dados
        for col in self.df.columns:
            quality_report['data_types'][col] = str(self.df[col].dtype)
        
        # Warnings específicos
        if 'risco_alto' not in self.df.columns:
            quality_report['warnings'].append("Variável target 'risco_alto' não encontrada")
        
        if 'data_ocorrencia' not in self.df.columns:
            quality_report['warnings'].append("Coluna de data não encontrada")
        
        if len(quality_report['warnings']) > 0:
            quality_report['status'] = 'warning'
        
        return quality_report
