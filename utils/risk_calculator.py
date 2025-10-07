"""
Calculadora de scores de risco operacional para policiais.
Implementa algoritmos para calcular risco baseado em múltiplos fatores.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OfficerRiskCalculator:
    """Calculadora de risco operacional para oficiais de polícia."""
    
    def __init__(self):
        """Inicializa a calculadora de risco."""
        
        # Pesos para diferentes fatores de risco
        self.risk_weights = {
            'weapon_risk': 0.30,      # Tipo de arma (maior peso)
            'crime_type_risk': 0.25,  # Tipo de crime
            'suspect_risk': 0.20,     # Número e perfil de suspeitos
            'location_risk': 0.15,    # Histórico do local
            'temporal_risk': 0.10     # Horário/dia da ocorrência
        }
        
        # Mapeamento de armas por nível de risco
        self.weapon_risk_map = {
            'Arma de Fogo': 1.0,
            'Explosivos': 1.0,
            'Faca': 0.6,
            'Objeto Contundente': 0.4,
            'Nenhum': 0.1,
            'Não Informado': 0.3
        }
        
        # Mapeamento de crimes por nível de risco
        self.crime_risk_map = {
            'Homicídio': 0.95,
            'Latrocínio': 0.95,
            'Sequestro': 0.90,
            'Tráfico de Drogas': 0.85,
            'Roubo': 0.80,
            'Estupro': 0.75,
            'Ameaça': 0.40,
            'Violência Doméstica': 0.60,
            'Furto': 0.30,
            'Estelionato': 0.20
        }
        
        # Turnos de maior risco (baseado em dados históricos)
        self.high_risk_shifts = {
            'Noite': 0.8,
            'Madrugada': 0.9,
            'Tarde': 0.6,
            'Manhã': 0.4
        }
    
    def calculate_weapon_risk(self, weapon):
        """
        Calcula risco baseado no tipo de arma.
        
        Args:
            weapon (str): Tipo de arma utilizada
            
        Returns:
            float: Score de risco da arma (0-1)
        """
        if pd.isna(weapon) or weapon == '':
            return self.weapon_risk_map['Não Informado']
        
        return self.weapon_risk_map.get(weapon, 0.3)
    
    def calculate_crime_type_risk(self, crime_type):
        """
        Calcula risco baseado no tipo de crime.
        
        Args:
            crime_type (str): Tipo do crime
            
        Returns:
            float: Score de risco do crime (0-1)
        """
        if pd.isna(crime_type) or crime_type == '':
            return 0.3  # Risco médio para crimes não informados
        
        return self.crime_risk_map.get(crime_type, 0.3)
    
    def calculate_suspect_risk(self, num_suspects, suspect_age=None, suspect_gender=None):
        """
        Calcula risco baseado no perfil dos suspeitos.
        
        Args:
            num_suspects (int): Número de suspeitos
            suspect_age (int): Idade do suspeito principal
            suspect_gender (str): Sexo do suspeito
            
        Returns:
            float: Score de risco dos suspeitos (0-1)
        """
        risk_score = 0.0
        
        # Risco por número de suspeitos
        if pd.isna(num_suspects):
            num_suspects = 1  # Assumir 1 se não informado
        
        if num_suspects == 0:
            risk_score += 0.1  # Suspeito em fuga
        elif num_suspects == 1:
            risk_score += 0.3
        elif num_suspects == 2:
            risk_score += 0.6
        elif num_suspects >= 3:
            risk_score += 0.9
        
        # Ajuste por idade (jovens podem ser mais impulsivos)
        if not pd.isna(suspect_age):
            if suspect_age <= 25:
                risk_score += 0.1
            elif suspect_age >= 50:
                risk_score -= 0.1
        
        # Manter score no range 0-1
        return min(max(risk_score, 0.0), 1.0)
    
    def calculate_location_risk(self, neighborhood, historical_data=None):
        """
        Calcula risco baseado no histórico do local.
        
        Args:
            neighborhood (str): Nome do bairro
            historical_data (pandas.DataFrame): Dados históricos do local
            
        Returns:
            float: Score de risco do local (0-1)
        """
        # Se não há dados históricos, usar risco médio
        if historical_data is None or neighborhood is None:
            return 0.5
        
        # Calcular taxa histórica de alto risco no bairro
        try:
            bairro_data = historical_data[historical_data['bairro'] == neighborhood]
            if len(bairro_data) == 0:
                return 0.5  # Sem dados históricos
            
            # Taxa de ocorrências de alto risco
            high_risk_rate = bairro_data['risco_alto'].mean()
            
            # Normalizar para 0-1 (assumindo que taxa máxima seria 0.5)
            location_risk = min(high_risk_rate * 2, 1.0)
            
            return location_risk
            
        except Exception:
            return 0.5  # Fallback para risco médio
    
    def calculate_temporal_risk(self, hour=None, day_of_week=None, shift=None):
        """
        Calcula risco baseado no momento da ocorrência.
        
        Args:
            hour (int): Hora da ocorrência (0-23)
            day_of_week (int): Dia da semana (0=segunda, 6=domingo)
            shift (str): Turno da ocorrência
            
        Returns:
            float: Score de risco temporal (0-1)
        """
        risk_score = 0.5  # Base
        
        # Risco por turno
        if shift and shift in self.high_risk_shifts:
            risk_score = self.high_risk_shifts[shift]
        
        # Ajuste por horário específico
        elif not pd.isna(hour):
            if 22 <= hour <= 23 or 0 <= hour <= 5:  # Madrugada/noite
                risk_score = 0.8
            elif 6 <= hour <= 11:  # Manhã
                risk_score = 0.3
            elif 12 <= hour <= 17:  # Tarde
                risk_score = 0.5
            else:  # Início da noite
                risk_score = 0.7
        
        # Ajuste por dia da semana
        if not pd.isna(day_of_week):
            if day_of_week in [4, 5, 6]:  # Sexta, sábado, domingo
                risk_score += 0.1
            
        return min(max(risk_score, 0.0), 1.0)
    
    def calculate_composite_risk_score(self, row, historical_data=None):
        """
        Calcula score composto de risco para uma ocorrência.
        
        Args:
            row (pandas.Series): Linha com dados da ocorrência
            historical_data (pandas.DataFrame): Dados históricos para contexto
            
        Returns:
            float: Score composto de risco (0-1)
        """
        try:
            # Calcular cada componente de risco
            weapon_risk = self.calculate_weapon_risk(row.get('arma_utilizada'))
            crime_risk = self.calculate_crime_type_risk(row.get('tipo_crime'))
            suspect_risk = self.calculate_suspect_risk(
                row.get('quantidade_suspeitos'),
                row.get('idade_suspeito'),
                row.get('sexo_suspeito')
            )
            location_risk = self.calculate_location_risk(
                row.get('bairro'), 
                historical_data
            )
            temporal_risk = self.calculate_temporal_risk(
                row.get('hora'),
                row.get('dia_semana_idx'),
                row.get('turno')
            )
            
            # Calcular score ponderado
            composite_score = (
                weapon_risk * self.risk_weights['weapon_risk'] +
                crime_risk * self.risk_weights['crime_type_risk'] +
                suspect_risk * self.risk_weights['suspect_risk'] +
                location_risk * self.risk_weights['location_risk'] +
                temporal_risk * self.risk_weights['temporal_risk']
            )
            
            return min(max(composite_score, 0.0), 1.0)
            
        except Exception:
            return 0.5  # Score médio em caso de erro
    
    def calculate_risk_scores(self, df):
        """
        Calcula scores de risco para todo o dataset.
        
        Args:
            df (pandas.DataFrame): DataFrame com dados das ocorrências
            
        Returns:
            pandas.DataFrame: DataFrame com scores de risco adicionados
        """
        df_with_scores = df.copy()
        
        # Calcular score composto para cada linha
        risk_scores = []
        for idx, row in df_with_scores.iterrows():
            score = self.calculate_composite_risk_score(row, df)
            risk_scores.append(score)
        
        df_with_scores['risk_score'] = risk_scores
        
        # Criar categorias de risco
        df_with_scores['risk_category'] = pd.cut(
            df_with_scores['risk_score'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Baixo', 'Médio', 'Alto'],
            include_lowest=True
        )
        
        return df_with_scores
    
    def get_risk_recommendations(self, risk_score, crime_type=None, weapon=None, num_suspects=None):
        """
        Gera recomendações táticas baseadas no score de risco.
        
        Args:
            risk_score (float): Score de risco (0-1)
            crime_type (str): Tipo do crime
            weapon (str): Tipo de arma
            num_suspects (int): Número de suspeitos
            
        Returns:
            dict: Recomendações táticas
        """
        recommendations = {
            'priority_level': 'Medium',
            'backup_required': False,
            'special_equipment': [],
            'tactical_notes': [],
            'response_protocol': 'Standard'
        }
        
        # Baseado no score de risco
        if risk_score >= 0.8:
            recommendations['priority_level'] = 'Critical'
            recommendations['backup_required'] = True
            recommendations['response_protocol'] = 'High Risk'
            recommendations['tactical_notes'].append('Situação de alto risco - extrema cautela')
            
        elif risk_score >= 0.6:
            recommendations['priority_level'] = 'High'
            recommendations['backup_required'] = True
            recommendations['response_protocol'] = 'Enhanced'
            recommendations['tactical_notes'].append('Risco elevado - solicitar backup')
            
        elif risk_score >= 0.4:
            recommendations['priority_level'] = 'Medium'
            recommendations['tactical_notes'].append('Risco moderado - manter protocolo padrão')
            
        else:
            recommendations['priority_level'] = 'Low'
            recommendations['tactical_notes'].append('Risco baixo - abordagem padrão')
        
        # Baseado no tipo de arma
        if weapon in ['Arma de Fogo', 'Explosivos']:
            recommendations['special_equipment'].extend(['Colete balístico', 'Escudo tático'])
            recommendations['tactical_notes'].append('ARMA LETAL PRESENTE - máxima cautela')
            recommendations['backup_required'] = True
            
        elif weapon == 'Faca':
            recommendations['special_equipment'].append('Escudo tático')
            recommendations['tactical_notes'].append('Arma branca - manter distância')
        
        # Baseado no número de suspeitos
        if num_suspects and num_suspects >= 3:
            recommendations['backup_required'] = True
            recommendations['tactical_notes'].append('Múltiplos suspeitos - necessário reforço')
        
        # Baseado no tipo de crime
        if crime_type in ['Homicídio', 'Latrocínio', 'Sequestro']:
            recommendations['response_protocol'] = 'High Risk'
            recommendations['backup_required'] = True
            recommendations['tactical_notes'].append('Crime violento - protocolo especial')
        
        return recommendations
    
    def analyze_risk_patterns(self, df):
        """
        Analisa padrões de risco no dataset.
        
        Args:
            df (pandas.DataFrame): DataFrame com scores de risco
            
        Returns:
            dict: Análise de padrões
        """
        if 'risk_score' not in df.columns:
            return {}
        
        analysis = {
            'overall_stats': {
                'avg_risk_score': df['risk_score'].mean(),
                'high_risk_percentage': (df['risk_score'] >= 0.7).mean() * 100,
                'low_risk_percentage': (df['risk_score'] <= 0.3).mean() * 100
            },
            'by_neighborhood': {},
            'by_time': {},
            'by_crime_type': {}
        }
        
        # Análise por bairro
        if 'bairro' in df.columns:
            bairro_analysis = df.groupby('bairro')['risk_score'].agg(['mean', 'count']).round(3)
            analysis['by_neighborhood'] = bairro_analysis.to_dict('index')
        
        # Análise temporal
        if 'turno' in df.columns:
            turno_analysis = df.groupby('turno')['risk_score'].agg(['mean', 'count']).round(3)
            analysis['by_time'] = turno_analysis.to_dict('index')
        
        # Análise por tipo de crime
        if 'tipo_crime' in df.columns:
            crime_analysis = df.groupby('tipo_crime')['risk_score'].agg(['mean', 'count']).round(3)
            analysis['by_crime_type'] = crime_analysis.to_dict('index')
        
        return analysis
    
    def get_high_risk_alerts(self, df, threshold=0.8):
        """
        Identifica alertas de alto risco.
        
        Args:
            df (pandas.DataFrame): DataFrame com scores de risco
            threshold (float): Threshold para alto risco
            
        Returns:
            pandas.DataFrame: Ocorrências de alto risco
        """
        if 'risk_score' not in df.columns:
            return pd.DataFrame()
        
        high_risk = df[df['risk_score'] >= threshold].copy()
        
        if len(high_risk) == 0:
            return pd.DataFrame()
        
        # Ordenar por score decrescente
        high_risk = high_risk.sort_values('risk_score', ascending=False)
        
        # Adicionar recomendações
        recommendations = []
        for _, row in high_risk.iterrows():
            rec = self.get_risk_recommendations(
                row['risk_score'],
                row.get('tipo_crime'),
                row.get('arma_utilizada'),
                row.get('quantidade_suspeitos')
            )
            recommendations.append(rec['tactical_notes'][0] if rec['tactical_notes'] else 'Alto risco')
        
        high_risk['tactical_recommendation'] = recommendations
        
        return high_risk
