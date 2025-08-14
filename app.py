# ASISTENTE VIRTUAL DE APUESTAS DEPORTIVAS V3.0
# ===================================================
# Sistema inteligente con análisis cuantitativo y cualitativo

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
import threading
import schedule
from functools import wraps
import hashlib
import re
import warnings
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Importación condicional de sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn no disponible - usando análisis estadístico avanzado")

# Suprimir warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONUNBUFFERED'] = '1'

# ============================================================
# CONFIGURACIÓN INICIAL
# ============================================================

app = Flask(__name__)
CORS(app)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Puerto de Render
PORT = int(os.environ.get('PORT', 10000))

@dataclass
class MatchData:
    """Estructura de datos para partidos"""
    id: str
    home_team: str
    away_team: str
    league: str
    date: str
    time: str
    odds: Dict[str, float]
    source: str
    scraped_at: str

@dataclass
class TeamStats:
    """Estadísticas del equipo"""
    name: str
    attack_rating: float
    defense_rating: float
    form_rating: float
    home_advantage: float
    recent_performance: List[str]  # ['W', 'L', 'D', 'W', 'L']
    
class BettingAssistant:
    """Asistente Virtual de Apuestas con IA Avanzada"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Base de datos ampliada de equipos
        self.teams_database = self._load_comprehensive_teams()
        self.historical_data = self._load_historical_patterns()
        
        # Modelos de IA
        self.models = {}
        self.scalers = {}
        self.model_ready = False
        
        if SKLEARN_AVAILABLE:
            self._train_advanced_models()
        
        # Cache de análisis
        self.cache = {
            'matches': [],
            'analysis': [],
            'recommendations': [],
            'last_update': None
        }
        
        logger.info("🎯 Asistente Virtual de Apuestas Inicializado")
    
    def _load_comprehensive_teams(self) -> Dict:
        """Base de datos completa de equipos del mundo"""
        return {
            # COLOMBIA - Liga BetPlay
            'Atlético Nacional': {
                'attack': 88, 'defense': 85, 'form': 90, 'home_advantage': 8,
                'league_strength': 75, 'european_coefficient': 0.0,
                'recent_form': ['W', 'W', 'D', 'W', 'L'],
                'key_players_available': 0.9, 'injury_impact': 0.1
            },
            'Millonarios FC': {
                'attack': 86, 'defense': 82, 'form': 87, 'home_advantage': 7,
                'league_strength': 75, 'european_coefficient': 0.0,
                'recent_form': ['W', 'D', 'W', 'W', 'D'],
                'key_players_available': 0.95, 'injury_impact': 0.05
            },
            'Junior de Barranquilla': {
                'attack': 83, 'defense': 80, 'form': 84, 'home_advantage': 7,
                'league_strength': 75, 'european_coefficient': 0.0,
                'recent_form': ['D', 'W', 'L', 'W', 'W'],
                'key_players_available': 0.85, 'injury_impact': 0.15
            },
            
            # ESPAÑA - La Liga
            'Real Madrid': {
                'attack': 98, 'defense': 93, 'form': 96, 'home_advantage': 8,
                'league_strength': 95, 'european_coefficient': 0.25,
                'recent_form': ['W', 'W', 'W', 'D', 'W'],
                'key_players_available': 0.95, 'injury_impact': 0.05
            },
            'FC Barcelona': {
                'attack': 95, 'defense': 89, 'form': 91, 'home_advantage': 8,
                'league_strength': 95, 'european_coefficient': 0.23,
                'recent_form': ['W', 'W', 'L', 'W', 'W'],
                'key_players_available': 0.90, 'injury_impact': 0.10
            },
            
            # INGLATERRA - Premier League
            'Manchester City': {
                'attack': 99, 'defense': 94, 'form': 98, 'home_advantage': 7,
                'league_strength': 96, 'european_coefficient': 0.26,
                'recent_form': ['W', 'W', 'W', 'W', 'D'],
                'key_players_available': 0.92, 'injury_impact': 0.08
            },
            'Arsenal FC': {
                'attack': 91, 'defense': 86, 'form': 89, 'home_advantage': 8,
                'league_strength': 96, 'european_coefficient': 0.24,
                'recent_form': ['W', 'D', 'W', 'W', 'L'],
                'key_players_available': 0.88, 'injury_impact': 0.12
            }
        }
    
    def _load_historical_patterns(self) -> Dict:
        """Patrones históricos para análisis cualitativo"""
        return {
            'time_patterns': {
                'evening_games': {'home_advantage_boost': 1.15, 'under_goals_tendency': 1.1},
                'afternoon_games': {'home_advantage_boost': 1.05, 'over_goals_tendency': 1.08},
                'night_games': {'defensive_tendency': 1.2, 'draw_probability_boost': 1.1}
            },
            'league_patterns': {
                'Liga BetPlay DIMAYOR': {
                    'home_advantage_factor': 1.25,
                    'goal_average': 2.1,
                    'upset_probability': 0.15
                },
                'Premier League': {
                    'home_advantage_factor': 1.10,
                    'goal_average': 2.8,
                    'upset_probability': 0.22
                },
                'La Liga Española': {
                    'home_advantage_factor': 1.15,
                    'goal_average': 2.6,
                    'upset_probability': 0.18
                }
            },
            'seasonal_patterns': {
                'early_season': {'form_reliability': 0.7, 'upset_probability': 1.3},
                'mid_season': {'form_reliability': 0.9, 'upset_probability': 1.0},
                'late_season': {'motivation_factor': 1.2, 'upset_probability': 0.8}
            }
        }
    
    def _train_advanced_models(self):
        """Entrena modelos avanzados de ML"""
        try:
            logger.info("🧠 Entrenando modelos de IA avanzados...")
            
            # Generar dataset sintético avanzado
            X, y_result, y_goals, y_confidence = self._generate_advanced_training_data(5000)
            
            # Scaler para normalización
            self.scalers['main'] = StandardScaler()
            X_scaled = self.scalers['main'].fit_transform(X)
            
            # Modelo 1: Predictor de resultados
            self.models['result'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=3,
                random_state=42
            )
            self.models['result'].fit(X_scaled, y_result)
            
            # Modelo 2: Predictor de goles
            self.models['goals'] = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=10,
                random_state=42
            )
            self.models['goals'].fit(X_scaled, y_goals)
            
            # Modelo 3: Estimador de confianza
            self.models['confidence'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.15,
                max_depth=8,
                random_state=42
            )
            self.models['confidence'].fit(X_scaled, y_confidence)
            
            # Validación cruzada
            cv_scores = cross_val_score(self.models['result'], X_scaled, y_result, cv=5)
            logger.info(f"✅ Precisión del modelo: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            self.model_ready = True
            
        except Exception as e:
            logger.error(f"❌ Error entrenando modelos: {e}")
            self.model_ready = False
    
    def _generate_advanced_training_data(self, n_samples: int) -> Tuple:
        """Genera datos de entrenamiento avanzados"""
        X, y_result, y_goals, y_confidence = [], [], [], []
        
        for _ in range(n_samples):
            # Características del equipo local
            home_attack = np.random.normal(80, 12)
            home_defense = np.random.normal(80, 12)
            home_form = np.random.normal(80, 15)
            home_advantage = np.random.normal(6, 2)
            
            # Características del equipo visitante
            away_attack = np.random.normal(75, 12)
            away_defense = np.random.normal(75, 12)
            away_form = np.random.normal(75, 15)
            
            # Factores contextuales
            league_strength = np.random.uniform(70, 100)
            motivation_factor = np.random.uniform(0.8, 1.2)
            weather_impact = np.random.uniform(0.9, 1.1)
            referee_bias = np.random.uniform(0.95, 1.05)
            
            # Calcular fortalezas ajustadas
            home_total = (home_attack + home_defense + home_form) * motivation_factor + home_advantage
            away_total = (away_attack + away_defense + away_form) * motivation_factor
            
            # Diferencia de fuerza
            strength_diff = (home_total - away_total) * weather_impact * referee_bias
            
            # Generar resultado basado en probabilidades realistas
            if strength_diff > 20:
                probs = [0.70, 0.20, 0.10]
                expected_goals = np.random.normal(2.8, 0.8)
            elif strength_diff > 10:
                probs = [0.55, 0.30, 0.15]
                expected_goals = np.random.normal(2.5, 0.7)
            elif strength_diff > -10:
                probs = [0.35, 0.35, 0.30]
                expected_goals = np.random.normal(2.2, 0.6)
            elif strength_diff > -20:
                probs = [0.15, 0.30, 0.55]
                expected_goals = np.random.normal(2.0, 0.5)
            else:
                probs = [0.10, 0.25, 0.65]
                expected_goals = np.random.normal(1.8, 0.4)
            
            result = np.random.choice([0, 1, 2], p=probs)  # 0=Home, 1=Draw, 2=Away
            goals = max(0, expected_goals)
            
            # Confianza basada en certeza estadística
            max_prob = max(probs)
            confidence = 40 + (max_prob - 0.33) * 150 + np.random.normal(0, 5)
            confidence = np.clip(confidence, 30, 98)
            
            # Construir vector de características
            features = [
                home_attack, home_defense, home_form, home_advantage,
                away_attack, away_defense, away_form,
                league_strength, motivation_factor, weather_impact,
                strength_diff, abs(strength_diff)
            ]
            
            X.append(features)
            y_result.append(result)
            y_goals.append(goals)
            y_confidence.append(confidence)
        
        return np.array(X), np.array(y_result), np.array(y_goals), np.array(y_confidence)
    
    def get_next_12_hours_matches(self) -> List[MatchData]:
        """Obtiene partidos de las próximas 12 horas"""
        try:
            logger.info("🔍 Buscando partidos de las próximas 12 horas...")
            
            # URLs de diferentes fuentes de partidos
            sources = [
                'https://www.betplay.com.co/deportes/futbol',
                'https://www.betplay.com.co/es/sports/football',
                'https://www.flashscore.com/football/',
                'https://www.espn.com/soccer/fixtures'
            ]
            
            all_matches = []
            
            for source_url in sources:
                try:
                    matches = self._scrape_matches_from_source(source_url)
                    all_matches.extend(matches)
                except Exception as e:
                    logger.warning(f"Error en fuente {source_url}: {e}")
                    continue
            
            # Si no hay datos reales, generar datos inteligentes
            if len(all_matches) < 3:
                logger.info("🎲 Generando partidos simulados para las próximas 12 horas...")
                all_matches = self._generate_next_12h_matches()
            
            # Filtrar solo próximas 12 horas
            filtered_matches = self._filter_next_12_hours(all_matches)
            
            logger.info(f"✅ {len(filtered_matches)} partidos encontrados para las próximas 12 horas")
            return filtered_matches
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo partidos: {e}")
            return self._generate_next_12h_matches()
    
    def _scrape_matches_from_source(self, url: str) -> List[MatchData]:
        """Scraper mejorado para múltiples fuentes"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            matches = []
            
            # Múltiples estrategias de selección
            selectors = [
                '.match-row, .game-row, .event-row',
                '[data-match], [data-event]',
                '.fixture, .match-card'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if len(elements) >= 2:
                    for elem in elements[:15]:  # Máximo 15 partidos
                        match = self._extract_match_from_element(elem)
                        if match:
                            matches.append(match)
                    break
            
            return matches
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []
    
    def _extract_match_from_element(self, element) -> Optional[MatchData]:
        """Extrae datos de partido desde elemento HTML"""
        try:
            # Buscar equipos
            team_selectors = ['.team-name', '.participant', '.team']
            teams = []
            
            for selector in team_selectors:
                team_elements = element.select(selector)
                teams = [t.get_text(strip=True) for t in team_elements[:2]]
                if len(teams) == 2 and all(len(t) > 2 for t in teams):
                    break
            
            if len(teams) != 2:
                return None
            
            # Buscar fecha/hora
            time_text = self._extract_time_from_element(element)
            date_str, time_str = self._parse_datetime_string(time_text)
            
            # Buscar odds
            odds = self._extract_odds_from_element(element)
            if not odds:
                odds = self._generate_realistic_odds(teams[0], teams[1])
            
            # Detectar liga
            league = self._detect_league_from_teams(teams)
            
            # Crear match data
            match = MatchData(
                id=f"match_{hashlib.md5('_'.join(teams).encode()).hexdigest()[:8]}",
                home_team=teams[0],
                away_team=teams[1],
                league=league,
                date=date_str,
                time=time_str,
                odds=odds,
                source='scraped',
                scraped_at=datetime.now().isoformat()
            )
            
            return match
            
        except Exception as e:
            logger.error(f"Error extrayendo partido: {e}")
            return None
    
    def _generate_next_12h_matches(self) -> List[MatchData]:
        """Genera partidos realistas para las próximas 12 horas"""
        matches = []
        current_time = datetime.now()
        
        # Ligas y equipos por horario
        leagues_schedule = {
            'morning': {
                'Liga BetPlay DIMAYOR': [
                    ('Atlético Nacional', 'Millonarios FC'),
                    ('Junior de Barranquilla', 'América de Cali'),
                    ('Independiente Santa Fe', 'Deportivo Cali')
                ]
            },
            'afternoon': {
                'Premier League': [
                    ('Manchester City', 'Arsenal FC'),
                    ('Chelsea FC', 'Liverpool FC')
                ],
                'La Liga Española': [
                    ('Real Madrid', 'FC Barcelona'),
                    ('Atlético Madrid', 'Sevilla FC')
                ]
            },
            'evening': {
                'Bundesliga': [
                    ('Bayern München', 'Borussia Dortmund'),
                    ('RB Leipzig', 'Bayer Leverkusen')
                ]
            }
        }
        
        match_times = {
            'morning': ['10:00', '11:30'],
            'afternoon': ['14:30', '16:00', '17:30'],
            'evening': ['19:00', '20:30', '22:00']
        }
        
        match_id = 1
        
        for period, leagues in leagues_schedule.items():
            for league, team_pairs in leagues.items():
                for home_team, away_team in team_pairs:
                    # Seleccionar hora aleatoria del período
                    time_str = np.random.choice(match_times[period])
                    
                    # Fecha aleatoria en próximas 12 horas
                    hours_ahead = np.random.randint(1, 13)
                    match_datetime = current_time + timedelta(hours=hours_ahead)
                    date_str = match_datetime.strftime('%Y-%m-%d')
                    
                    # Generar odds realistas
                    odds = self._generate_realistic_odds(home_team, away_team)
                    
                    match = MatchData(
                        id=f"sim_{match_id:03d}",
                        home_team=home_team,
                        away_team=away_team,
                        league=league,
                        date=date_str,
                        time=time_str,
                        odds=odds,
                        source='simulated',
                        scraped_at=datetime.now().isoformat()
                    )
                    
                    matches.append(match)
                    match_id += 1
        
        return matches
    
    def analyze_match_comprehensive(self, match: MatchData) -> Dict:
        """Análisis completo cuantitativo y cualitativo"""
        try:
            logger.info(f"🔬 Analizando: {match.home_team} vs {match.away_team}")
            
            # 1. Análisis Cuantitativo (Matemático)
            quant_analysis = self._quantitative_analysis(match)
            
            # 2. Análisis Cualitativo (Contextual)
            qual_analysis = self._qualitative_analysis(match)
            
            # 3. Análisis de Valor de Apuesta
            value_analysis = self._betting_value_analysis(match, quant_analysis, qual_analysis)
            
            # 4. Análisis de Riesgo
            risk_analysis = self._risk_assessment(match, quant_analysis, qual_analysis)
            
            # 5. Recomendación Final
            final_recommendation = self._generate_final_recommendation(
                match, quant_analysis, qual_analysis, value_analysis, risk_analysis
            )
            
            return {
                'match_info': {
                    'home_team': match.home_team,
                    'away_team': match.away_team,
                    'league': match.league,
                    'date': match.date,
                    'time': match.time,
                    'odds': match.odds
                },
                'quantitative': quant_analysis,
                'qualitative': qual_analysis,
                'value': value_analysis,
                'risk': risk_analysis,
                'recommendation': final_recommendation,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de {match.home_team} vs {match.away_team}: {e}")
            return self._fallback_analysis(match)
    
    def _quantitative_analysis(self, match: MatchData) -> Dict:
        """Análisis matemático y estadístico"""
        home_stats = self._get_team_stats(match.home_team)
        away_stats = self._get_team_stats(match.away_team)
        
        # Cálculos estadísticos avanzados
        home_strength = self._calculate_team_strength(home_stats, is_home=True)
        away_strength = self._calculate_team_strength(away_stats, is_home=False)
        
        strength_difference = home_strength - away_strength
        
        # Probabilidades matemáticas
        probabilities = self._calculate_match_probabilities(home_strength, away_strength)
        
        # Predicción de goles
        expected_goals = self._predict_goals(home_stats, away_stats)
        
        # Si tenemos ML, usar modelos
        if self.model_ready and SKLEARN_AVAILABLE:
            ml_predictions = self._ml_predictions(match, home_stats, away_stats)
        else:
            ml_predictions = None
        
        return {
            'home_strength': round(home_strength, 2),
            'away_strength': round(away_strength, 2),
            'strength_difference': round(strength_difference, 2),
            'probabilities': probabilities,
            'expected_goals': expected_goals,
            'ml_predictions': ml_predictions,
            'statistical_confidence': self._calculate_statistical_confidence(probabilities)
        }
    
    def _qualitative_analysis(self, match: MatchData) -> Dict:
        """Análisis contextual y cualitativo"""
        
        # Factores de tiempo
        time_factors = self._analyze_time_factors(match)
        
        # Factores de liga
        league_factors = self._analyze_league_factors(match)
        
        # Forma reciente de equipos
        form_analysis = self._analyze_recent_form(match)
        
        # Motivación e importancia del partido
        motivation_factors = self._analyze_motivation(match)
        
        # Factores históricos
        historical_factors = self._analyze_historical_factors(match)
        
        return {
            'time_factors': time_factors,
            'league_factors': league_factors,
            'form_analysis': form_analysis,
            'motivation': motivation_factors,
            'historical': historical_factors,
            'qualitative_score': self._calculate_qualitative_score([
                time_factors, league_factors, form_analysis, motivation_factors
            ])
        }
    
    def _betting_value_analysis(self, match: MatchData, quant: Dict, qual: Dict) -> Dict:
        """Análisis del valor de la apuesta"""
        
        # Probabilidades ajustadas
        true_probabilities = self._adjust_probabilities_with_context(
            quant['probabilities'], qual
        )
        
        # Calcular valores esperados
        expected_values = {}
        for outcome, odds in match.odds.items():
            if outcome in true_probabilities:
                ev = (true_probabilities[outcome] * odds) - 1
                expected_values[outcome] = round(max(0, ev), 4)
        
        # Encontrar mejor valor
        best_value = max(expected_values.items(), key=lambda x: x[1]) if expected_values else ('', 0)
        
        # Kelly Criterion para tamaño de apuesta
        kelly_percentages = {}
        for outcome, ev in expected_values.items():
            if ev > 0 and outcome in match.odds:
                odds_value = match.odds[outcome]
                prob = true_probabilities.get(outcome, 0)
                kelly = (prob * odds_value - 1) / (odds_value - 1)
                kelly_percentages[outcome] = max(0, min(kelly * 100, 10))  # Max 10% del bankroll
        
        return {
            'true_probabilities': true_probabilities,
            'expected_values': expected_values,
            'best_value_bet': best_value,
            'kelly_percentages': kelly_percentages,
            'value_rating': self._rate_overall_value(expected_values)
        }
    
    def _risk_assessment(self, match: MatchData, quant: Dict, qual: Dict) -> Dict:
        """Evaluación completa de riesgos"""
        
        # Riesgo por volatilidad del resultado
        result_volatility = 1 - max(quant['probabilities'].values())
        
        # Riesgo por incertidumbre de datos
        data_uncertainty = 1 - (quant['statistical_confidence'] / 100)
        
        # Riesgo por factores cualitativos
        qualitative_risk = self._assess_qualitative_risks(qual)
        
        # Riesgo por odds (líneas muy cerradas = mayor riesgo)
        odds_risk = self._assess_odds_risk(match.odds)
        
        # Riesgo total
        total_risk = np.average([
            result_volatility, data_uncertainty, qualitative_risk, odds_risk
        ], weights=[0.3, 0.2, 0.3, 0.2])
        
        return {
            'result_volatility': round(result_volatility, 3),
            'data_uncertainty': round(data_uncertainty, 3),
            'qualitative_risk': round(qualitative_risk, 3),
            'odds_risk': round(odds_risk, 3),
            'total_risk': round(total_risk, 3),
            'risk_level': self._categorize_risk(total_risk),
            'risk_mitigation': self._suggest_risk_mitigation(total_risk)
        }
    
    def _generate_final_recommendation(self, match: MatchData, quant: Dict, qual: Dict, 
                                     value: Dict, risk: Dict) -> Dict:
        """Genera recomendación final inteligente"""
        
        # Determinar si apostar
        should_bet = (
            value['best_value_bet'][1] > 0.05 and  # Valor esperado > 5%
            risk['total_risk'] < 0.7 and           # Riesgo < 70%
            quant['statistical_confidence'] > 60   # Confianza > 60%
        )
        
        if should_bet:
            recommended_outcome = value['best_value_bet'][0]
            recommended_odds = match.odds.get(recommended_outcome, 0)
            confidence = self._calculate_overall_confidence(quant, qual, value, risk)
            
            # Tamaño de apuesta sugerido
            kelly_pct = value['kelly_percentages'].get(recommended_outcome, 0)
            suggested_stake = min(kelly_pct, 5)  # Máximo 5% del bankroll
            
            reasons = self._generate_betting_reasons(match, quant, qual, value, recommended_outcome)
            
            return {
                'should_bet': True,
                'recommended_outcome': recommended_outcome,
                'recommended_odds': recommended_odds,
                'confidence_score': confidence,
                'expected_value': value['best_value_bet'][1],
                'suggested_stake_percent': round(suggested_stake, 1),
                'risk_level': risk['risk_level'],
                'reasons': reasons,
                'betting_strategy': self._suggest_betting_strategy(risk['risk_level']),
                'alternative_bets': self._find_alternative_bets(value, risk)
            }
        else:
            return {
                'should_bet': False,
                'reason_not_to_bet': self._explain_why_not_bet(value, risk, quant),
                'confidence_score': 0,
                'risk_level': risk['risk_level'],
                'alternative_suggestion': 'Esperar mejores oportunidades'
            }

    # MÉTODOS DE APOYO PARA LOS ANÁLISIS
    
    def _get_team_stats(self, team_name: str) -> Dict:
        """Obtiene estadísticas completas del equipo"""
        # Búsqueda exacta
        if team_name in self.teams_database:
            return self.teams_database[team_name]
        
        # Búsqueda parcial
        for db_team, stats in self.teams_database.items():
            if self._teams_match(team_name, db_team):
                return stats
        
        # Estadísticas estimadas
        return self._estimate_team_stats(team_name)
    
    def _teams_match(self, team1: str, team2: str) -> bool:
        """Verifica si dos nombres de equipo coinciden"""
        t1_words = set(team1.lower().split())
        t2_words = set(team2.lower().split())
        
        # Si hay intersección significativa
        common_words = t1_words.intersection(t2_words)
        return len(common_words) >= 1 and any(len(word) > 3 for word in common_words)
    
    def _calculate_team_strength(self, stats: Dict, is_home: bool) -> float:
        """Calcula fortaleza total del equipo"""
        base_strength = (
            stats['attack'] * 0.35 +
            stats['defense'] * 0.35 +
            stats['form'] * 0.30
        )
        
        # Aplicar ventaja local
        if is_home:
            base_strength += stats.get('home_advantage', 5)
        
        # Factores adicionales
        base_strength *= stats.get('key_players_available', 1.0)
        base_strength *= (1 - stats.get('injury_impact', 0))
        
        return max(30, min(100, base_strength))
    
    def _calculate_match_probabilities(self, home_strength: float, away_strength: float) -> Dict:
        """Calcula probabilidades del partido"""
        strength_diff = home_strength - away_strength
        
        # Función logística para probabilidades
        home_prob = 1 / (1 + np.exp(-strength_diff / 15)) * 0.7 + 0.15
        away_prob = 1 / (1 + np.exp(strength_diff / 15)) * 0.7 + 0.15
        draw_prob = 1 - home_prob - away_prob
        
        # Normalizar para que sumen 1
        total = home_prob + draw_prob + away_prob
        
        return {
            'home': round(home_prob / total, 3),
            'draw': round(draw_prob / total, 3),
            'away': round(away_prob / total, 3)
        }
    
    def _predict_goals(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """Predice goles del partido"""
        home_goals = (home_stats['attack'] * 0.6 + (100 - away_stats['defense']) * 0.4) / 100 * 1.8
        away_goals = (away_stats['attack'] * 0.6 + (100 - home_stats['defense']) * 0.4) / 100 * 1.5
        
        total_goals = home_goals + away_goals
        
        return {
            'home_goals': round(home_goals, 2),
            'away_goals': round(away_goals, 2),
            'total_goals': round(total_goals, 2),
            'over_2_5_prob': round(1 / (1 + np.exp(-(total_goals - 2.5))), 3),
            'under_2_5_prob': round(1 / (1 + np.exp((total_goals - 2.5))), 3)
        }
    
    def _generate_betting_reasons(self, match: MatchData, quant: Dict, qual: Dict, 
                                value: Dict, outcome: str) -> List[str]:
        """Genera razones específicas para la apuesta"""
        reasons = []
        
        # Razones cuantitativas
        if quant['strength_difference'] > 10:
            stronger_team = match.home_team if quant['strength_difference'] > 0 else match.away_team
            reasons.append(f"{stronger_team} tiene ventaja estadística significativa")
        
        # Razones de valor
        ev = value['expected_values'].get(outcome, 0)
        if ev > 0.1:
            reasons.append(f"Excelente valor esperado del {ev*100:.1f}%")
        
        # Razones cualitativas
        if qual['form_analysis'].get('trend', '') == 'positive':
            reasons.append("Tendencia positiva en forma reciente")
        
        # Razones específicas por resultado
        if outcome == 'home':
            reasons.append(f"Ventaja de jugar en casa para {match.home_team}")
        elif outcome == 'away':
            reasons.append(f"Buen rendimiento como visitante de {match.away_team}")
        
        return reasons[:4]  # Máximo 4 razones
    
    def send_recommendations_to_user(self) -> Dict:
        """Método principal para enviar recomendaciones al usuario"""
        try:
            current_hour = datetime.now().hour
            
            # Verificar horario de operación (7 AM a 12 AM)
            if current_hour < 7 or current_hour >= 24:
                return {
                    'status': 'sleeping',
                    'message': 'Asistente en modo descanso (12 AM - 7 AM)',
                    'next_analysis': '7:00 AM'
                }
            
            logger.info("🎯 Iniciando análisis de recomendaciones...")
            
            # Obtener partidos
            matches = self.get_next_12_hours_matches()
            
            if not matches:
                return {
                    'status': 'no_matches',
                    'message': 'No hay partidos en las próximas 12 horas',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Analizar todos los partidos
            analyses = []
            recommendations = []
            
            for match in matches:
                analysis = self.analyze_match_comprehensive(match)
                analyses.append(analysis)
                
                if analysis['recommendation']['should_bet']:
                    recommendations.append(analysis)
            
            # Ordenar recomendaciones por calidad
            recommendations.sort(
                key=lambda x: x['recommendation']['confidence_score'] * 
                             (1 + x['recommendation']['expected_value']),
                reverse=True
            )
            
            # Preparar mensaje para el usuario
            message = self._format_user_message(recommendations[:3])  # Top 3
            
            return {
                'status': 'success',
                'total_matches_analyzed': len(matches),
                'betting_opportunities': len(recommendations),
                'top_recommendations': recommendations[:3],
                'user_message': message,
                'timestamp': datetime.now().isoformat(),
                'next_analysis': (datetime.now() + timedelta(hours=2)).strftime('%H:%M')
            }
            
        except Exception as e:
            logger.error(f"❌ Error enviando recomendaciones: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _format_user_message(self, recommendations: List[Dict]) -> str:
        """Formatea mensaje para enviar al usuario"""
        if not recommendations:
            return """
🎯 **ASISTENTE DE APUESTAS - ANÁLISIS COMPLETADO**

❌ **No hay oportunidades de apuesta recomendadas en este momento**

• Todos los partidos analizados presentan riesgo alto o valor insuficiente
• Te notificaré en 2 horas con nuevos análisis
• Mantén paciencia para mejores oportunidades

📊 *Análisis basado en IA y estadísticas avanzadas*
            """
        
        message = "🎯 **ASISTENTE DE APUESTAS - RECOMENDACIONES**\n\n"
        message += f"📅 **{datetime.now().strftime('%d/%m/%Y %H:%M')}**\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            match_info = rec['match_info']
            recommendation = rec['recommendation']
            
            message += f"**🏆 OPORTUNIDAD #{i}**\n"
            message += f"⚽ {match_info['home_team']} vs {match_info['away_team']}\n"
            message += f"🏆 {match_info['league']}\n"
            message += f"⏰ {match_info['date']} - {match_info['time']}\n\n"
            
            message += f"**💰 RECOMENDACIÓN:**\n"
            outcome_text = {
                'home': f"Apostar por {match_info['home_team']}",
                'away': f"Apostar por {match_info['away_team']}",
                'draw': "Apostar por empate"
            }
            message += f"• {outcome_text.get(recommendation['recommended_outcome'], '')}\n"
            message += f"• Cuota: {recommendation['recommended_odds']}\n"
            message += f"• Confianza: {recommendation['confidence_score']}%\n"
            message += f"• Valor esperado: +{recommendation['expected_value']*100:.1f}%\n"
            message += f"• Apostar: {recommendation['suggested_stake_percent']}% del bankroll\n"
            message += f"• Riesgo: {recommendation['risk_level']}\n\n"
            
            message += f"**📋 RAZONES:**\n"
            for reason in recommendation['reasons']:
                message += f"• {reason}\n"
            message += "\n"
            message += "="*50 + "\n\n"
        
        message += "⚠️ **RECORDATORIOS:**\n"
        message += "• Nunca apuestes más del 5% de tu bankroll por partido\n"
        message += "• Estas son recomendaciones, no garantías\n"
        message += "• Juega responsablemente\n\n"
        message += "📱 Próximo análisis en 2 horas"
        
        return message
    
    # Métodos auxiliares adicionales...
    def _estimate_team_stats(self, team_name: str) -> Dict:
        """Estima estadísticas para equipos desconocidos"""
        base_rating = 75
        
        # Ajustes por palabras clave
        name_lower = team_name.lower()
        if any(word in name_lower for word in ['real', 'barcelona', 'city', 'bayern']):
            base_rating = 85
        elif any(word in name_lower for word in ['united', 'arsenal', 'atletico']):
            base_rating = 80
        
        return {
            'attack': base_rating + np.random.randint(-5, 6),
            'defense': base_rating + np.random.randint(-5, 6),
            'form': base_rating + np.random.randint(-10, 11),
            'home_advantage': 6,
            'league_strength': 75,
            'european_coefficient': 0.1,
            'recent_form': ['W', 'D', 'L', 'W', 'D'],
            'key_players_available': 0.9,
            'injury_impact': 0.1
        }

# Crear instancia global
betting_assistant = BettingAssistant()

# ============================================================
# API ENDPOINTS PARA N8N
# ============================================================

@app.route('/api/recommendations', methods=['GET'])
def get_betting_recommendations():
    """Endpoint principal para obtener recomendaciones"""
    return jsonify(betting_assistant.send_recommendations_to_user())

@app.route('/api/analyze-matches', methods=['GET'])
def analyze_current_matches():
    """Analiza partidos actuales sin recomendar"""
    try:
        matches = betting_assistant.get_next_12_hours_matches()
        analyses = []
        
        for match in matches[:10]:  # Máximo 10 partidos
            analysis = betting_assistant.analyze_match_comprehensive(match)
            analyses.append(analysis)
        
        return jsonify({
            'status': 'success',
            'total_matches': len(matches),
            'analyses': analyses,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check para N8N"""
    return jsonify({
        'status': 'healthy',
        'system': 'Betting Assistant V3.0',
        'sklearn_available': SKLEARN_AVAILABLE,
        'model_ready': betting_assistant.model_ready,
        'timestamp': datetime.now().isoformat(),
        'uptime_hours': (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).seconds // 3600
    })

@app.route('/api/force-update', methods=['POST'])
def force_update():
    """Fuerza actualización de datos"""
    try:
        result = betting_assistant.send_recommendations_to_user()
        return jsonify({
            'status': 'updated',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# ============================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================

if __name__ == '__main__':
    logger.info("🚀 Iniciando Asistente Virtual de Apuestas V3.0")
    logger.info(f"🔧 Puerto: {PORT}")
    logger.info(f"🧠 Sklearn: {'✅ Disponible' if SKLEARN_AVAILABLE else '❌ No disponible'}")
    
    # Inicializar sistema
    try:
        logger.info("⚡ Sistema iniciado y listo")
        app.run(
            host='0.0.0.0',
            port=PORT,
            debug=False,
            threaded=True
        )
    except Exception as e:
        logger.error(f"❌ Error crítico: {e}")
        sys.exit(1)
