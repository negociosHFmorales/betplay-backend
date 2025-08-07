@app.route('/api/analysis')
def get_analysis():
    """Endpoint para análisis estadístico avanzado"""
    try:
        matches = CACHE.get('analysis_results', [])
        
        if not matches:
            return jsonify({
                'status': 'warning', 
                'message': 'No hay datos disponibles para análisis',
                'timestamp': datetime.now().isoformat()
            })
        
        # Extraer análisis
        analyses = [m.get('analysis', {}) for m in matches if 'analysis' in m]
        
        if not analyses:
            return jsonify({
                'status': 'warning',
                'message': 'No hay análisis disponibles',
                'timestamp': datetime.now().isoformat()
            })
        
        # Estadísticas generales
        confidences = [a.get('confidence', 0) for a in analyses]
        expected_values = [a.get('expected_value', 0) for a in analyses]
        recommendations = [a.get('recommendation', '') for a in analyses]
        
        # Calcular métricas
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        high_confidence = len([c for c in confidences if c >= 75])
        medium_confidence = len([c for c in confidences if 60 <= c < 75])
        low_confidence = len([c for c in confidences if c < 60])
        
        positive_value = len([ev for ev in expected_values if ev > 0])
        avg_expected_value = sum([ev for ev in expected_values if ev > 0]) / len([ev for ev in expected_values if ev > 0]) if any(ev > 0 for ev in expected_values) else 0
        
        # Distribución de recomendaciones
        rec_distribution = {
            'home': recommendations.count('home'),
            'draw': recommendations.count('draw'),
            'away': recommendations.count('away')
        }
        
        # Mejores oportunidades
        valid_matches = [m for m in matches if 'analysis' in m and m['analysis'].get('expected_value', 0) > 0]
        best_opportunities = sorted(
            valid_matches,
            key=lambda x: x['analysis']['confidence'] * (1 + x['analysis']['expected_value']) * 100,
            reverse=True
        )[:10]
        
        # Análisis por liga
        league_analysis = {}
        for match in matches:
            league = match.get('league', 'Desconocida')
            if league not in league_analysis:
                league_analysis[league] = {
                    'matches': 0,
                    'avg_confidence': 0,
                    'total_confidence': 0
                }
            league_analysis[league]['matches'] += 1
            if 'analysis' in match:
                league_analysis[league]['total_confidence'] += match['analysis'].get('confidence', 0)
        
        for league, data in league_analysis.items():
            if data['matches'] > 0:
                data['avg_confidence'] = round(data['total_confidence'] / data['matches'], 1)
            del data['total_confidence']  # Limpiar dato temporal
        
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_matches': len(matches),
                'analyzed_matches': len(analyses),
                'average_confidence': round(avg_confidence, 1),
                'confidence_distribution': {
                    'high': high_confidence,
                    'medium': medium_confidence,
                    'low': low_confidence
                },
                'positive_expected_value': positive_value,
                'average_expected_value': round(avg_expected_value, 3),
                'recommendation_distribution': rec_distribution,
                'last_update': CACHE.get('last_update').isoformat() if CACHE.get('last_update') else None
            },
            'best_opportunities': [
                {
                    'match': f"{m['home_team']} vs {m['away_team']}",
                    'league': m.get('league', ''),
                    'confidence': m['analysis']['confidence'],
                    'expected_value': m['analysis']['expected_value'],
                    'recommendation': m['analysis']['recommendation'],
                    'risk_level': m['analysis'].get('risk_level', 'MEDIO'),
                    'win_probability': m['analysis'].get('win_probability', 0)
                } for m in best_opportunities
            ],
            'league_analysis': league_analysis,
            'system_info': {
                'sklearn_available': SKLEARN_AVAILABLE,
                'analysis_type': 'ML' if SKLEARN_AVAILABLE else 'Statistical'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en /api/analysis: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/recommendations')
def get_recommendations():
    """Endpoint para mejores recomendaciones de apuesta"""
    try:
        matches = CACHE.get('analysis_results', [])
        valid_matches = [m for m in matches if 'analysis' in m]
        
        if not valid_matches:
            return jsonify({
                'status': 'warning',
                'message': 'No hay recomendaciones disponibles',
                'timestamp': datetime.now().isoformat()
            })
        
        # Filtrar mejores apuestas (criterios estrictos)
        premium_bets = []
        good_bets = []
        moderate_bets = []
        
        for match in valid_matches:
            analysis = match['analysis']
            confidence = analysis.get('confidence', 0)
            expected_value = analysis.get('expected_value', 0)
            risk_level = analysis.get('risk_level', 'ALTO')
            
            bet_data = {
                'match': f"{match['home_team']} vs {match['away_team']}",
                'league': match.get('league', ''),
                'date': match.get('date', ''),
                'time': match.get('time', ''),
                'recommendation': f"Apostar por {analysis['recommendation']}",
                'confidence': confidence,
                'expected_value': expected_value,
                'win_probability': analysis.get('win_probability', 0),
                'reasons': analysis.get('reasons', []),
                'risk_level': risk_level,
                'analysis_type': analysis.get('analysis_type', 'Statistical'),
                'score': confidence * (1 + expected_value) * 100  # Score combinado
            }
            
            # Categorizar apuestas
            if confidence >= 80 and expected_value > 0.08 and risk_level in ['BAJO', 'MEDIO-BAJO']:
                premium_bets.append(bet_data)
            elif confidence >= 70 and expected_value > 0.05:
                good_bets.append(bet_data)
            elif confidence >= 60 and expected_value > 0.02:
                moderate_bets.append(bet_data)
        
        # Ordenar por score
        premium_bets.sort(key=lambda x: x['score'], reverse=True)
        good_bets.sort(key=lambda x: x['score'], reverse=True)
        moderate_bets.sort(key=lambda x: x['score'], reverse=True)
        
        # Limitar resultados
        premium_bets = premium_bets[:5]
        good_bets = good_bets[:8]
        moderate_bets = moderate_bets[:10]
        
        # Todas las recomendaciones ordenadas por score
        all_recommendations = premium_bets + good_bets + moderate_bets
        all_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_matches_analyzed': len(valid_matches),
                'premium_opportunities': len(premium_bets),
                'good_opportunities': len(good_bets),
                'moderate_opportunities': len(moderate_bets),
                'recommendation_criteria': {
                    'premium': 'Confianza ≥80%, Valor esperado >8%, Riesgo bajo',
                    'good': 'Confianza ≥70%, Valor esperado >5%',
                    'moderate': 'Confianza ≥60%, Valor esperado >2%'
                }
            },
            'recommendations': {
                'premium': premium_bets,
                'good': good_bets,
                'moderate': moderate_bets,
                'all': all_recommendations[:15]  # Top 15 general
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en /api/recommendations: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/update', methods=['POST'])
def force_update():
    """Endpoint para forzar actualización manual"""
    try:
        # Verificar si el sistema está inicializado
        if not CACHE.get('initialization_complete'):
            return jsonify({
                'status': 'error',
                'message': 'Sistema no inicializado completamente',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        logger.info("🔄 Actualización manual solicitada")
        start_time = time.time()
        
        # Realizar actualización
        success = update_betting_data()
        duration = time.time() - start_time
        
        if success:
            matches_analyzed = len(CACHE.get('analysis_results', []))
            matches_scraped = len(CACHE.get('scraped_data', []))
            
            return jsonify({
                'status': 'success',
                'message': f'Actualización completada en {duration:.2f} segundos',
                'data': {
                    'matches_scraped': matches_scraped,
                    'matches_analyzed': matches_analyzed,
                    'success_rate': f"{(matches_analyzed/matches_scraped*100):.1f}%" if matches_scraped > 0 else "0%",
                    'error_count': CACHE.get('error_count', 0)
                },
                'timestamp': datetime.now().isoformat(),
                'next_auto_update': (datetime.now() + timedelta(seconds=CACHE['update_frequency'])).isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Error durante la actualización',
                'duration': f"{duration:.2f} segundos",
                'timestamp': datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        logger.error(f"Error en actualización manual: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error inesperado: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health')
def health_check():
    """Endpoint de health check detallado"""
    try:
        current_time = datetime.now()
        
        # Información del sistema
        system_info = {
            'status': 'healthy' if CACHE.get('initialization_complete') else 'initializing',
            'timestamp': current_time.isoformat(),
            'uptime_seconds': int((current_time - CACHE['system_start_time']).total_seconds()),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'port': PORT
        }
        
        # Estado de componentes
        components = {
            'scraper': 'ok' if scraper else 'not_initialized',
            'analyzer': 'ok' if analyzer else 'not_initialized',
            'sklearn': 'available' if SKLEARN_AVAILABLE else 'not_available',
            'cache': 'ok',
            'api': 'ok'
        }
        
        # Métricas de datos
        data_metrics = {
            'scraped_matches': len(CACHE.get('scraped_data', [])),
            'analyzed_matches': len(CACHE.get('analysis_results', [])),
            'last_update': CACHE.get('last_update').isoformat() if CACHE.get('last_update') else None,
            'error_count': CACHE.get('error_count', 0),
            'needs_update': should_update_data()
        }
        
        # Calcular tiempo desde última actualización
        if CACHE.get('last_update'):
            time_since_update = current_time - CACHE['last_update']
            data_metrics['hours_since_update'] = round(time_since_update.total_seconds() / 3600, 2)
        else:
            data_metrics['hours_since_update'] = None
        
        # Estado general
        overall_status = 'healthy'
        if not CACHE.get('initialization_complete'):
            overall_status = 'initializing'
        elif CACHE.get('error_count', 0) > 5:
            overall_status = 'degraded'
        elif not CACHE.get('analysis_results'):
            overall_status = 'warning'
        
        response = {
            'status': overall_status,
            'system': system_info,
            'components': components,
            'data': data_metrics
        }
        
        # Código de estado HTTP basado en el estado general
        status_code = 200
        if overall_status == 'initializing':
            status_code = 503
        
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# ============================================================
# SCHEDULER Y TAREAS AUTOMÁTICAS
# ============================================================

def schedule_automatic_updates():
    """Configura actualizaciones automáticas"""
    schedule.every(3).hours.do(update_betting_data)
    logger.info("📅 Actualizaciones automáticas programadas cada 3 horas")
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Verificar cada minuto
    
    # Ejecutar scheduler en hilo separado
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("✅ Scheduler iniciado en hilo separado")

# ============================================================
# MANEJO DE ERRORES
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint no encontrado',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno del servidor: {error}")
    return jsonify({
        'status': 'error',
        'message': 'Error interno del servidor',
        'timestamp': datetime.now().isoformat()
    }), 500

# ============================================================
# INICIALIZACIÓN Y PUNTO DE ENTRADA
# ============================================================

def startup_sequence():
    """Secuencia de inicio del sistema"""
    logger.info("🌟 Iniciando BetPlay Colombia - Sistema de Análisis IA")
    logger.info(f"🐍 Python {sys.version}")
    logger.info(f"🔢 Puerto: {PORT}")
    logger.info(f"🧠 Sklearn disponible: {SKLEARN_AVAILABLE}")
    
    # Inicializar sistema
    success = initialize_system()
    if success:
        logger.info("✅ Inicialización exitosa")
        
        # Programar actualizaciones automáticas
        schedule_automatic_updates()
        
        return True
    else:
        logger.error("❌ Error en inicialización")
        return False

# ============================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================

if __name__ == '__main__':
    try:
        # Ejecutar secuencia de inicio
        startup_success = startup_sequence()
        
        if startup_success:
            logger.info(f"🚀 Iniciando servidor Flask en puerto {PORT}")
            logger.info("🌐 Dashboard disponible en: http://localhost:10000")
            logger.info("📡 API disponible en: http://localhost:10000/api/")
            
            # Iniciar servidor Flask
            app.run(
                host='0.0.0.0',
                port=PORT,
                debug=False,  # Desactivar debug en producción
                threaded=True,
                use_reloader=False  # Evitar doble inicialización
            )
        else:
            logger.error("❌ No se pudo inicializar el sistema")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error crítico en punto de entrada: {e}")
        sys.exit(1)# BETPLAY COLOMBIA - SCRAPER & ANALYZER BACKEND V2.1
# ===================================================
# Versión optimizada y corregida para Render

from flask import Flask, jsonify, render_template_string, request
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

# Importación condicional de sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn no disponible - usando análisis simplificado")

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

# Configuración de BetPlay
BETPLAY_CONFIG = {
    'base_url': 'https://www.betplay.com.co',
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'es-CO,es;q=0.9,en;q=0.8',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache'
    },
    'timeout': 8,
    'max_retries': 2
}

# Cache global
CACHE = {
    'scraped_data': [],
    'analysis_results': [],
    'last_update': None,
    'update_frequency': 3 * 3600,  # 3 horas
    'system_start_time': datetime.now(),
    'initialization_complete': False,
    'error_count': 0
}

# ============================================================
# SCRAPER OPTIMIZADO
# ============================================================

class BetPlayScraper:
    """Scraper robusto para BetPlay"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(BETPLAY_CONFIG['headers'])
        
    def _make_request(self, url, timeout=None):
        """Petición HTTP con retry"""
        timeout = timeout or BETPLAY_CONFIG['timeout']
        
        for attempt in range(BETPLAY_CONFIG['max_retries']):
            try:
                logger.info(f"Intentando conectar: {url} (intento {attempt + 1})")
                response = self.session.get(url, timeout=timeout)
                
                if response.status_code == 200:
                    logger.info(f"✅ Conexión exitosa - {len(response.content)} bytes")
                    return response
                else:
                    logger.warning(f"HTTP {response.status_code} en intento {attempt + 1}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout en intento {attempt + 1}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error de conexión en intento {attempt + 1}: {str(e)[:100]}")
            
            if attempt < BETPLAY_CONFIG['max_retries'] - 1:
                time.sleep(2 ** attempt)  # Backoff exponencial
        
        logger.error(f"❌ Falló conexión a {url} después de {BETPLAY_CONFIG['max_retries']} intentos")
        return None
    
    def scrape_soccer_matches(self):
        """Scraper principal con múltiples estrategias"""
        try:
            logger.info("🔍 Iniciando scraping de partidos...")
            
            # Estrategia 1: URLs principales de BetPlay
            primary_urls = [
                f"{BETPLAY_CONFIG['base_url']}/deportes/futbol",
                f"{BETPLAY_CONFIG['base_url']}/es/sports/football",
                f"{BETPLAY_CONFIG['base_url']}/apuestas/deportes"
            ]
            
            for url in primary_urls:
                matches = self._try_scrape_url(url)
                if matches and len(matches) > 0:
                    logger.info(f"✅ {len(matches)} partidos obtenidos de {url}")
                    return matches
            
            # Estrategia 2: Datos simulados inteligentes
            logger.info("🎲 Usando datos simulados inteligentes")
            return self._generate_smart_mock_data()
            
        except Exception as e:
            logger.error(f"❌ Error en scraping: {e}")
            CACHE['error_count'] += 1
            return self._generate_smart_mock_data()
    
    def _try_scrape_url(self, url):
        """Intenta hacer scraping de una URL"""
        response = self._make_request(url)
        if not response:
            return []
        
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            matches = self._parse_matches_advanced(soup)
            return matches
        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return []
    
    def _parse_matches_advanced(self, soup):
        """Parser avanzado con múltiples selectores"""
        matches = []
        
        # Múltiples selectores para diferentes layouts
        selectors_list = [
            # Selectores específicos de BetPlay
            '.event-row, .match-row, .game-row',
            '.betting-event, .sport-event, .match-item',
            '.fixture, .match-card, .event-card',
            # Selectores generales
            '[data-match], [data-event], [data-game]',
            '.team-name',  # Fallback básico
        ]
        
        for selectors in selectors_list:
            containers = soup.select(selectors)
            logger.info(f"Encontrados {len(containers)} elementos con selector: {selectors}")
            
            if containers and len(containers) >= 2:
                for container in containers[:20]:  # Límite de seguridad
                    match = self._extract_match_data(container)
                    if match and self._is_valid_match(match):
                        matches.append(match)
                
                if len(matches) >= 3:  # Mínimo viable
                    break
        
        # Limpiar duplicados
        unique_matches = []
        seen_pairs = set()
        for match in matches:
            pair = (match['home_team'], match['away_team'])
            if pair not in seen_pairs and pair[::-1] not in seen_pairs:
                seen_pairs.add(pair)
                unique_matches.append(match)
        
        logger.info(f"Partidos únicos extraídos: {len(unique_matches)}")
        return unique_matches
    
    def _extract_match_data(self, container):
        """Extractor robusto de datos de partido"""
        try:
            # Estrategias para encontrar equipos
            teams = self._find_teams_multiple_strategies(container)
            if not teams or len(teams) < 2:
                return None
            
            # ID único del partido
            match_id = f"bp_{hashlib.md5(f'{teams[0]}_{teams[1]}'.encode()).hexdigest()[:8]}"
            
            # Fecha y hora inteligente
            date_info = self._extract_date_time(container)
            
            # Odds inteligentes
            odds = self._extract_odds(container) or self._generate_realistic_odds(teams)
            
            # Liga inteligente
            league = self._detect_league(teams)
            
            return {
                'id': match_id,
                'sport': 'soccer',
                'league': league,
                'home_team': teams[0],
                'away_team': teams[1],
                'date': date_info['date'],
                'time': date_info['time'],
                'odds': odds,
                'source': 'betplay',
                'scraped_at': datetime.now().isoformat(),
                'confidence_score': self._calculate_extraction_confidence(teams, odds)
            }
            
        except Exception as e:
            logger.error(f"Error extrayendo datos: {e}")
            return None
    
    def _find_teams_multiple_strategies(self, container):
        """Múltiples estrategias para encontrar equipos"""
        strategies = [
            # Estrategia 1: Selectores específicos
            lambda c: [elem.get_text(strip=True) for elem in c.select('.team-name, .participant-name')[:2]],
            # Estrategia 2: Por atributos
            lambda c: [elem.get('title', '').strip() for elem in c.select('[title]') if elem.get('title', '').strip()][:2],
            # Estrategia 3: Texto directo
            lambda c: [elem.get_text(strip=True) for elem in c.select('span, div') if len(elem.get_text(strip=True).split()) <= 3 and len(elem.get_text(strip=True)) > 3][:2],
            # Estrategia 4: Por clases comunes
            lambda c: [elem.get_text(strip=True) for elem in c.select('.home, .away')][:2],
        ]
        
        for strategy in strategies:
            try:
                teams = strategy(container)
                if teams and len(teams) >= 2 and all(len(team.strip()) > 2 for team in teams):
                    # Limpiar nombres
                    clean_teams = [self._clean_team_name(team) for team in teams[:2]]
                    if all(clean_teams):
                        return clean_teams
            except:
                continue
        
        return None
    
    def _clean_team_name(self, name):
        """Limpia nombres de equipos"""
        if not name:
            return ""
        
        # Remover caracteres especiales
        name = re.sub(r'[^\w\s\.\-]', ' ', name)
        # Remover espacios extra
        name = ' '.join(name.split())
        # Capitalizar
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name.strip() if 2 < len(name.strip()) < 50 else ""
    
    def _extract_date_time(self, container):
        """Extrae fecha y hora del partido"""
        # Buscar elementos de fecha/hora
        time_selectors = ['.match-time, .event-time', '[data-time]', '.time, .fecha']
        
        for selector in time_selectors:
            elements = container.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if text and any(char.isdigit() for char in text):
                    # Procesar fecha/hora encontrada
                    return self._parse_datetime(text)
        
        # Valores por defecto inteligentes
        now = datetime.now()
        return {
            'date': (now + timedelta(days=np.random.randint(0, 3))).strftime('%Y-%m-%d'),
            'time': f"{np.random.randint(15, 22)}:{np.random.choice(['00', '30'])}"
        }
    
    def _parse_datetime(self, text):
        """Parse inteligente de fecha/hora"""
        # Patrones comunes
        patterns = [
            r'(\d{1,2}):(\d{2})',  # HH:MM
            r'(\d{1,2})/(\d{1,2})',  # DD/MM
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                # Procesar según el patrón
                if ':' in pattern:  # Es hora
                    hour, minute = match.groups()
                    return {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'time': f"{hour.zfill(2)}:{minute}"
                    }
        
        # Default
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': f"{np.random.randint(15, 22)}:00"
        }
    
    def _extract_odds(self, container):
        """Extrae odds del container"""
        try:
            # Selectores para odds
            odds_selectors = ['.odds, .odd, .quota', '[data-odd]', '.price']
            
            odds_values = []
            for selector in odds_selectors:
                elements = container.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    # Buscar números decimales
                    match = re.search(r'(\d+\.?\d*)', text)
                    if match:
                        value = float(match.group(1))
                        if 1.1 <= value <= 20.0:  # Rango válido de odds
                            odds_values.append(value)
            
            if len(odds_values) >= 3:
                return {
                    'home': round(odds_values[0], 2),
                    'draw': round(odds_values[1], 2),
                    'away': round(odds_values[2], 2)
                }
        except:
            pass
        
        return None
    
    def _generate_realistic_odds(self, teams):
        """Genera odds realistas basadas en los equipos"""
        home_strength = self._estimate_team_strength(teams[0])
        away_strength = self._estimate_team_strength(teams[1])
        
        # Ajustar por ventaja local
        home_strength += 5
        
        diff = home_strength - away_strength
        
        if diff > 15:
            return {'home': round(np.random.uniform(1.5, 1.9), 2), 'draw': round(np.random.uniform(3.5, 4.2), 2), 'away': round(np.random.uniform(4.0, 8.0), 2)}
        elif diff > 5:
            return {'home': round(np.random.uniform(1.8, 2.3), 2), 'draw': round(np.random.uniform(3.0, 3.7), 2), 'away': round(np.random.uniform(2.8, 4.5), 2)}
        elif diff < -15:
            return {'home': round(np.random.uniform(4.0, 8.0), 2), 'draw': round(np.random.uniform(3.5, 4.2), 2), 'away': round(np.random.uniform(1.5, 1.9), 2)}
        elif diff < -5:
            return {'home': round(np.random.uniform(2.8, 4.5), 2), 'draw': round(np.random.uniform(3.0, 3.7), 2), 'away': round(np.random.uniform(1.8, 2.3), 2)}
        else:
            return {'home': round(np.random.uniform(2.4, 3.0), 2), 'draw': round(np.random.uniform(3.0, 3.5), 2), 'away': round(np.random.uniform(2.4, 3.0), 2)}
    
    def _estimate_team_strength(self, team_name):
        """Estima la fuerza del equipo por nombre"""
        # Equipos top colombianos
        top_colombian = ['nacional', 'millonarios', 'junior', 'america']
        # Equipos top internacionales
        top_international = ['real madrid', 'barcelona', 'city', 'bayern', 'psg']
        
        team_lower = team_name.lower()
        
        if any(top in team_lower for top in top_international):
            return np.random.randint(85, 95)
        elif any(top in team_lower for top in top_colombian):
            return np.random.randint(75, 85)
        else:
            return np.random.randint(65, 80)
    
    def _detect_league(self, teams):
        """Detecta la liga basada en los equipos"""
        team_text = ' '.join(teams).lower()
        
        if any(word in team_text for word in ['nacional', 'millonarios', 'junior', 'america', 'santa fe', 'medellin']):
            return 'Liga BetPlay DIMAYOR'
        elif any(word in team_text for word in ['real madrid', 'barcelona', 'atletico', 'sevilla']):
            return 'La Liga Española'
        elif any(word in team_text for word in ['city', 'united', 'arsenal', 'chelsea', 'liverpool']):
            return 'Premier League'
        elif any(word in team_text for word in ['bayern', 'dortmund', 'leipzig']):
            return 'Bundesliga'
        elif any(word in team_text for word in ['psg', 'marseille', 'lyon']):
            return 'Ligue 1'
        else:
            return 'Liga Internacional'
    
    def _is_valid_match(self, match):
        """Valida si un partido es válido"""
        if not match:
            return False
        
        required_fields = ['home_team', 'away_team', 'odds']
        if not all(field in match for field in required_fields):
            return False
        
        if len(match['home_team']) < 3 or len(match['away_team']) < 3:
            return False
        
        if match['home_team'] == match['away_team']:
            return False
        
        return True
    
    def _calculate_extraction_confidence(self, teams, odds):
        """Calcula confianza de extracción"""
        score = 0
        
        # Confianza en teams
        if all(len(team) > 5 for team in teams):
            score += 30
        elif all(len(team) > 3 for team in teams):
            score += 20
        
        # Confianza en odds
        if odds and all(1.1 <= odd <= 10.0 for odd in odds.values()):
            score += 40
        elif odds:
            score += 20
        
        # Bonus por completitud
        if len(teams) == 2 and odds:
            score += 30
        
        return min(score, 100)
    
    def _generate_smart_mock_data(self):
        """Genera datos simulados inteligentes y realistas"""
        logger.info("🎲 Generando datos simulados inteligentes...")
        
        # Base de datos de equipos reales
        teams_data = {
            'colombia': {
                'teams': ['Atlético Nacional', 'Millonarios FC', 'Junior de Barranquilla', 'América de Cali', 
                         'Independiente Santa Fe', 'Deportivo Independiente Medellín', 'Once Caldas', 'Deportivo Cali'],
                'league': 'Liga BetPlay DIMAYOR',
                'strength_range': (70, 85)
            },
            'spain': {
                'teams': ['Real Madrid', 'FC Barcelona', 'Atlético Madrid', 'Sevilla FC'],
                'league': 'La Liga Española',
                'strength_range': (85, 95)
            },
            'england': {
                'teams': ['Manchester City', 'Arsenal FC', 'Manchester United', 'Chelsea FC'],
                'league': 'Premier League',
                'strength_range': (82, 92)
            },
            'germany': {
                'teams': ['Bayern München', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen'],
                'league': 'Bundesliga',
                'strength_range': (80, 90)
            }
        }
        
        matches = []
        
        for country, data in teams_data.items():
            teams = data['teams'].copy()
            np.random.shuffle(teams)
            
            # Generar 2-3 partidos por liga
            num_matches = np.random.randint(2, 4)
            for i in range(0, min(len(teams)-1, num_matches*2), 2):
                home_team = teams[i]
                away_team = teams[i+1]
                
                # Calcular fortalezas
                home_strength = np.random.randint(*data['strength_range']) + 5  # Ventaja local
                away_strength = np.random.randint(*data['strength_range'])
                
                # Generar odds realistas
                odds = self._calculate_realistic_odds(home_strength, away_strength)
                
                # Fecha inteligente
                match_date = datetime.now() + timedelta(days=np.random.randint(0, 4))
                match_time = f"{np.random.randint(15, 22)}:{np.random.choice(['00', '30'])}"
                
                match = {
                    'id': f'smart_{country}_{i//2}',
                    'sport': 'soccer',
                    'league': data['league'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'date': match_date.strftime('%Y-%m-%d'),
                    'time': match_time,
                    'odds': odds,
                    'source': 'betplay_smart',
                    'scraped_at': datetime.now().isoformat(),
                    'confidence_score': 85,
                    'is_simulated': True
                }
                
                matches.append(match)
        
        logger.info(f"✅ {len(matches)} partidos simulados generados inteligentemente")
        return matches
    
    def _calculate_realistic_odds(self, home_strength, away_strength):
        """Calcula odds realistas basadas en fortaleza"""
        diff = home_strength - away_strength
        
        # Probabilidades basadas en diferencia de fuerza
        if diff > 20:
            prob_home, prob_draw, prob_away = 0.65, 0.25, 0.10
        elif diff > 10:
            prob_home, prob_draw, prob_away = 0.55, 0.28, 0.17
        elif diff > 0:
            prob_home, prob_draw, prob_away = 0.45, 0.30, 0.25
        elif diff > -10:
            prob_home, prob_draw, prob_away = 0.35, 0.30, 0.35
        elif diff > -20:
            prob_home, prob_draw, prob_away = 0.25, 0.28, 0.47
        else:
            prob_home, prob_draw, prob_away = 0.15, 0.25, 0.60
        
        # Convertir a odds (incluyendo margen de casa)
        margin = 0.05  # 5% margen
        odds_home = round(1 / (prob_home * (1 - margin)), 2)
        odds_draw = round(1 / (prob_draw * (1 - margin)), 2)
        odds_away = round(1 / (prob_away * (1 - margin)), 2)
        
        return {
            'home': max(1.1, min(odds_home, 15.0)),
            'draw': max(1.1, min(odds_draw, 15.0)),
            'away': max(1.1, min(odds_away, 15.0))
        }

# ============================================================
# ANALIZADOR CON IA
# ============================================================

class BettingAnalyzer:
    """Analizador inteligente con o sin sklearn"""
    
    def __init__(self):
        self.sklearn_available = SKLEARN_AVAILABLE
        self.models = {}
        self.scalers = {}
        self.team_database = self._load_comprehensive_team_database()
        self.models_ready = False
        
        if self.sklearn_available:
            logger.info("✅ Sklearn disponible - Usando ML")
            self._train_ml_models()
        else:
            logger.info("ℹ️  Sklearn no disponible - Usando análisis estadístico")
            self.models_ready = True
    
    def _load_comprehensive_team_database(self):
        """Base de datos comprensiva de equipos"""
        return {
            # Liga Colombiana
            'Atlético Nacional': {'attack': 86, 'defense': 82, 'form': 88, 'home_advantage': 8},
            'Millonarios FC': {'attack': 84, 'defense': 80, 'form': 85, 'home_advantage': 7},
            'Junior de Barranquilla': {'attack': 81, 'defense': 78, 'form': 82, 'home_advantage': 7},
            'América de Cali': {'attack': 78, 'defense': 79, 'form': 76, 'home_advantage': 6},
            'Independiente Santa Fe': {'attack': 75, 'defense': 78, 'form': 74, 'home_advantage': 6},
            'Deportivo Independiente Medellín': {'attack': 79, 'defense': 76, 'form': 78, 'home_advantage': 6},
            'Once Caldas': {'attack': 72, 'defense': 74, 'form': 71, 'home_advantage': 5},
            'Deportivo Cali': {'attack': 74, 'defense': 72, 'form': 70, 'home_advantage': 5},
            
            # España - La Liga
            'Real Madrid': {'attack': 96, 'defense': 91, 'form': 94, 'home_advantage': 8},
            'FC Barcelona': {'attack': 93, 'defense': 87, 'form': 89, 'home_advantage': 8},
            'Atlético Madrid': {'attack': 85, 'defense': 90, 'form': 87, 'home_advantage': 7},
            'Sevilla FC': {'attack': 82, 'defense': 84, 'form': 81, 'home_advantage': 6},
            
            # Inglaterra - Premier League
            'Manchester City': {'attack': 98, 'defense': 92, 'form': 96, 'home_advantage': 7},
            'Arsenal FC': {'attack': 89, 'defense': 84, 'form': 88, 'home_advantage': 8},
            'Manchester United': {'attack': 85, 'defense': 80, 'form': 82, 'home_advantage': 8},
            'Chelsea FC': {'attack': 87, 'defense': 83, 'form': 84, 'home_advantage': 7},
            
            # Alemania - Bundesliga
            'Bayern München': {'attack': 95, 'defense': 90, 'form': 93, 'home_advantage': 8},
            'Borussia Dortmund': {'attack': 88, 'defense': 81, 'form': 85, 'home_advantage': 9},
            'RB Leipzig': {'attack': 84, 'defense': 83, 'form': 82, 'home_advantage': 6},
            'Bayer Leverkusen': {'attack': 86, 'defense': 79, 'form': 83, 'home_advantage': 6},
        }
    
    def _train_ml_models(self):
        """Entrena modelos ML si sklearn está disponible"""
        try:
            logger.info("🧠 Entrenando modelos de Machine Learning...")
            
            # Generar dataset sintético pero realista
            np.random.seed(42)
            n_samples = 2000
            
            X, y = self._generate_training_data(n_samples)
            
            # Entrenar modelo de clasificación (resultado)
            self.scalers['result'] = StandardScaler()
            X_scaled = self.scalers['result'].fit_transform(X)
            
            self.models['result'] = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                random_state=42
            )
            self.models['result'].fit(X_scaled, y)
            
            # Entrenar modelo de confianza
            confidence_data = self._generate_confidence_data(n_samples)
            self.models['confidence'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            
            X_conf, y_conf = confidence_data
            X_conf_scaled = self.scalers['result'].transform(X_conf)
            self.models['confidence'].fit(X_conf_scaled, y_conf)
            
            self.models_ready = True
            logger.info("✅ Modelos ML entrenados exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error entrenando modelos ML: {e}")
            self.sklearn_available = False
            self.models_ready = True
    
    def _generate_training_data(self, n_samples):
        """Genera datos de entrenamiento realistas"""
        X = []
        y = []
        
        for _ in range(n_samples):
            # Estadísticas de equipos
            home_attack = np.random.normal(75, 15)
            home_defense = np.random.normal(75, 15)
            home_form = np.random.normal(75, 20)
            away_attack = np.random.normal(75, 15)
            away_defense = np.random.normal(75, 15)
            away_form = np.random.normal(75, 20)
            
            # Limitar valores
            stats = [max(30, min(100, x)) for x in [home_attack, home_defense, home_form, away_attack, away_defense, away_form]]
            
            # Calcular fortalezas
            home_strength = (stats[0] + stats[1] + stats[2]) / 3 + np.random.normal(7, 2)  # Ventaja local
            away_strength = (stats[3] + stats[4] + stats[5]) / 3
            
            # Diferencia de fuerza
            strength_diff = home_strength - away_strength
            
            # Calcular probabilidades realistas
            if strength_diff > 15:
                probs = [0.60, 0.25, 0.15]
            elif strength_diff > 5:
                probs = [0.50, 0.30, 0.20]
            elif strength_diff > -5:
                probs = [0.35, 0.30, 0.35]
            elif strength_diff > -15:
                probs = [0.20, 0.30, 0.50]
            else:
                probs = [0.15, 0.25, 0.60]
            
            # Generar resultado
            result = np.random.choice([0, 1, 2], p=probs)  # 0=Local, 1=Empate, 2=Visitante
            
            X.append(stats + [strength_diff])
            y.append(result)
        
        return np.array(X), np.array(y)
    
    def _generate_confidence_data(self, n_samples):
        """Genera datos para modelo de confianza"""
        X, y_result = self._generate_training_data(n_samples)
        
        # Confianza basada en diferencia de fuerza y consistencia
        y_confidence = []
        for i, features in enumerate(X):
            strength_diff = abs(features[-1])  # Diferencia absoluta
            form_consistency = abs(features[2] - features[5])  # Diferencia de forma
            
            base_confidence = 50
            base_confidence += min(strength_diff * 2, 30)  # Más diferencia = más confianza
            base_confidence -= min(form_consistency, 15)  # Menos consistencia = menos confianza
            base_confidence += np.random.normal(0, 5)  # Ruido
            
            y_confidence.append(max(40, min(95, base_confidence)))
        
        return X, np.array(y_confidence)
    
    def analyze_match(self, match_data):
        """Análisis principal del partido"""
        try:
            home_team = match_data.get('home_team', '')
            away_team = match_data.get('away_team', '')
            odds = match_data.get('odds', {})
            league = match_data.get('league', '')
            
            # Obtener estadísticas
            home_stats = self._get_team_stats(home_team)
            away_stats = self._get_team_stats(away_team)
            
            if self.sklearn_available and self.models_ready:
                return self._analyze_with_ml(home_team, away_team, home_stats, away_stats, odds, league)
            else:
                return self._analyze_statistical(home_team, away_team, home_stats, away_stats, odds, league)
            
        except Exception as e:
            logger.error(f"❌ Error analizando partido {home_team} vs {away_team}: {e}")
            return self._fallback_analysis(match_data)
    
    def _get_team_stats(self, team_name):
        """Obtiene estadísticas del equipo"""
        # Busqueda exacta
        if team_name in self.team_database:
            return self.team_database[team_name]
        
        # Búsqueda parcial
        for db_team, stats in self.team_database.items():
            if any(word in team_name.lower() for word in db_team.lower().split()):
                return stats
            if any(word in db_team.lower() for word in team_name.lower().split()):
                return stats
        
        # Stats por defecto basados en el nombre
        base_strength = self._estimate_strength_by_name(team_name)
        return {
            'attack': base_strength + np.random.randint(-5, 6),
            'defense': base_strength + np.random.randint(-5, 6),
            'form': base_strength + np.random.randint(-10, 11),
            'home_advantage': 6
        }
    
    def _estimate_strength_by_name(self, team_name):
        """Estima fuerza por nombre del equipo"""
        name_lower = team_name.lower()
        
        # Palabras clave para equipos fuertes
        strong_keywords = ['real', 'barcelona', 'city', 'united', 'bayern', 'nacional', 'millonarios']
        medium_keywords = ['atletico', 'arsenal', 'chelsea', 'junior', 'america']
        
        if any(keyword in name_lower for keyword in strong_keywords):
            return np.random.randint(80, 90)
        elif any(keyword in name_lower for keyword in medium_keywords):
            return np.random.randint(72, 82)
        else:
            return np.random.randint(65, 78)
    
    def _analyze_with_ml(self, home_team, away_team, home_stats, away_stats, odds, league):
        """Análisis usando Machine Learning"""
        try:
            # Preparar features
            features = [
                home_stats['attack'],
                home_stats['defense'],
                home_stats['form'],
                away_stats['attack'],
                away_stats['defense'],
                away_stats['form']
            ]
            
            # Calcular diferencia de fuerza
            home_strength = sum(features[:3]) / 3 + home_stats['home_advantage']
            away_strength = sum(features[3:]) / 3
            strength_diff = home_strength - away_strength
            
            features_array = np.array(features + [strength_diff]).reshape(1, -1)
            features_scaled = self.scalers['result'].transform(features_array)
            
            # Predicción de resultado
            result_probs = self.models['result'].predict_proba(features_scaled)[0]
            predicted_result = np.argmax(result_probs)
            
            # Predicción de confianza
            confidence = self.models['confidence'].predict(features_scaled)[0]
            
            # Mapear resultado a recomendación
            result_map = {0: 'home', 1: 'draw', 2: 'away'}
            recommendation = result_map[predicted_result]
            
            # Calcular valor esperado
            rec_odds = odds.get(recommendation, 2.0)
            win_probability = result_probs[predicted_result]
            expected_value = (win_probability * rec_odds) - 1
            
            # Generar razones inteligentes
            reasons = self._generate_intelligent_reasons(
                home_team, away_team, home_stats, away_stats, 
                strength_diff, win_probability, league
            )
            
            return {
                'confidence': round(max(45, min(95, confidence)), 1),
                'recommendation': recommendation,
                'expected_value': round(max(0, expected_value), 3),
                'win_probability': round(win_probability * 100, 1),
                'home_strength': round(home_strength, 1),
                'away_strength': round(away_strength, 1),
                'reasons': reasons,
                'risk_level': self._calculate_risk_level(confidence, expected_value),
                'analysis_type': 'ML',
                'model_confidence': round(confidence, 1)
            }
            
        except Exception as e:
            logger.error(f"Error en análisis ML: {e}")
            return self._analyze_statistical(home_team, away_team, home_stats, away_stats, odds, league)
    
    def _analyze_statistical(self, home_team, away_team, home_stats, away_stats, odds, league):
        """Análisis estadístico sin ML"""
        try:
            # Calcular fortalezas
            home_strength = (home_stats['attack'] + home_stats['defense'] + home_stats['form']) / 3
            home_strength += home_stats['home_advantage']  # Ventaja local
            
            away_strength = (away_stats['attack'] + away_stats['defense'] + away_stats['form']) / 3
            
            strength_diff = home_strength - away_strength
            
            # Calcular probabilidades basadas en diferencia
            if strength_diff > 20:
                home_prob, draw_prob, away_prob = 0.65, 0.25, 0.10
                confidence = 85
            elif strength_diff > 10:
                home_prob, draw_prob, away_prob = 0.55, 0.30, 0.15
                confidence = 78
            elif strength_diff > 3:
                home_prob, draw_prob, away_prob = 0.48, 0.32, 0.20
                confidence = 70
            elif strength_diff > -3:
                home_prob, draw_prob, away_prob = 0.35, 0.35, 0.30
                confidence = 65
            elif strength_diff > -10:
                home_prob, draw_prob, away_prob = 0.25, 0.32, 0.43
                confidence = 72
            elif strength_diff > -20:
                home_prob, draw_prob, away_prob = 0.15, 0.30, 0.55
                confidence = 80
            else:
                home_prob, draw_prob, away_prob = 0.10, 0.25, 0.65
                confidence = 87
            
            # Determinar recomendación
            probs = {'home': home_prob, 'draw': draw_prob, 'away': away_prob}
            recommendation = max(probs, key=probs.get)
            win_probability = probs[recommendation]
            
            # Calcular valor esperado
            rec_odds = odds.get(recommendation, 2.0)
            expected_value = (win_probability * rec_odds) - 1
            
            # Ajustar confianza por factores adicionales
            if abs(home_stats['form'] - away_stats['form']) > 15:
                confidence += 5
            if league and 'betplay' in league.lower():
                confidence += 3  # Conocimiento local
            
            # Añadir variabilidad realista
            confidence += np.random.randint(-3, 4)
            confidence = max(50, min(95, confidence))
            
            # Generar razones
            reasons = self._generate_intelligent_reasons(
                home_team, away_team, home_stats, away_stats,
                strength_diff, win_probability, league
            )
            
            return {
                'confidence': round(confidence, 1),
                'recommendation': recommendation,
                'expected_value': round(max(0, expected_value), 3),
                'win_probability': round(win_probability * 100, 1),
                'home_strength': round(home_strength, 1),
                'away_strength': round(away_strength, 1),
                'reasons': reasons,
                'risk_level': self._calculate_risk_level(confidence, expected_value),
                'analysis_type': 'Statistical',
                'strength_difference': round(strength_diff, 1)
            }
            
        except Exception as e:
            logger.error(f"Error en análisis estadístico: {e}")
            return self._fallback_analysis({'home_team': home_team, 'away_team': away_team})
    
    def _generate_intelligent_reasons(self, home_team, away_team, home_stats, away_stats, strength_diff, win_prob, league):
        """Genera razones inteligentes para la recomendación"""
        reasons = []
        
        # Razón principal por diferencia de fuerza
        if strength_diff > 15:
            reasons.append(f"{home_team} muestra superioridad clara en todas las estadísticas")
        elif strength_diff > 5:
            reasons.append(f"Ligera ventaja para {home_team} como local")
        elif strength_diff < -15:
            reasons.append(f"{away_team} es considerablemente superior")
        elif strength_diff < -5:
            reasons.append(f"{away_team} supera las estadísticas del local")
        else:
            reasons.append("Encuentro equilibrado con resultado incierto")
        
        # Razones específicas por estadísticas
        if home_stats['form'] > away_stats['form'] + 10:
            reasons.append(f"{home_team} atraviesa un mejor momento de forma")
        elif away_stats['form'] > home_stats['form'] + 10:
            reasons.append(f"{away_team} llega en mejor estado de forma")
        
        if home_stats['attack'] > 85:
            reasons.append(f"Potente ataque de {home_team}")
        elif away_stats['attack'] > 85:
            reasons.append(f"Gran poder ofensivo de {away_team}")
        
        if home_stats['defense'] > 85:
            reasons.append(f"Sólida defensa local de {home_team}")
        elif away_stats['defense'] > 85:
            reasons.append(f"Defensa visitante muy confiable de {away_team}")
        
        # Razones por probabilidad
        if win_prob > 0.6:
            reasons.append("Alta probabilidad matemática según el análisis")
        elif win_prob < 0.4:
            reasons.append("Resultado con menor certeza estadística")
        
        # Razones por liga
        if league and 'betplay' in league.lower():
            reasons.append("Análisis especializado en fútbol colombiano")
        elif league and any(word in league.lower() for word in ['premier', 'la liga', 'bundesliga']):
            reasons.append("Liga de alto nivel competitivo")
        
        # Limitar a las 4 mejores razones
        return reasons[:4]
    
    def _calculate_risk_level(self, confidence, expected_value):
        """Calcula nivel de riesgo"""
        if confidence >= 80 and expected_value > 0.1:
            return 'BAJO'
        elif confidence >= 70 and expected_value > 0.05:
            return 'MEDIO-BAJO'
        elif confidence >= 60 and expected_value > 0:
            return 'MEDIO'
        elif confidence >= 50:
            return 'MEDIO-ALTO'
        else:
            return 'ALTO'
    
    def _fallback_analysis(self, match_data):
        """Análisis de respaldo básico"""
        home_team = match_data.get('home_team', 'Local')
        away_team = match_data.get('away_team', 'Visitante')
        
        # Análisis básico aleatorio pero coherente
        confidence = np.random.randint(55, 75)
        recommendations = ['home', 'draw', 'away']
        recommendation = np.random.choice(recommendations, p=[0.4, 0.3, 0.3])
        
        return {
            'confidence': confidence,
            'recommendation': recommendation,
            'expected_value': round(np.random.uniform(0, 0.08), 3),
            'win_probability': np.random.randint(35, 55),
            'home_strength': np.random.randint(70, 80),
            'away_strength': np.random.randint(70, 80),
            'reasons': [
                'Análisis básico por disponibilidad limitada',
                f'Evaluación estándar para {home_team} vs {away_team}'
            ],
            'risk_level': 'MEDIO',
            'analysis_type': 'Fallback'
        }

# ============================================================
# SISTEMA PRINCIPAL Y CACHE
# ============================================================

# Instancias globales
scraper = None
analyzer = None

def initialize_system():
    """Inicialización completa del sistema"""
    global scraper, analyzer
    
    try:
        logger.info("🚀 Inicializando BetPlay Colombia System...")
        
        # Inicializar componentes
        scraper = BetPlayScraper()
        analyzer = BettingAnalyzer()
        
        logger.info("✅ Componentes inicializados")
        
        # Primera actualización de datos
        logger.info("📊 Realizando primera carga de datos...")
        success = update_betting_data()
        
        if success:
            CACHE['initialization_complete'] = True
            logger.info("✅ Sistema completamente inicializado y operativo")
        else:
            logger.warning("⚠️  Sistema inicializado con advertencias")
            CACHE['initialization_complete'] = True  # Permitir operación
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error crítico en inicialización: {e}")
        CACHE['initialization_complete'] = False
        return False

def update_betting_data():
    """Actualización principal de datos"""
    global scraper, analyzer
    
    if not scraper or not analyzer:
        logger.error("❌ Sistema no inicializado")
        return False
    
    try:
        update_start_time = time.time()
        logger.info("🔄 Iniciando actualización de datos...")
        
        # Scraping de partidos
        logger.info("📥 Obteniendo partidos...")
        matches = scraper.scrape_soccer_matches()
        
        if not matches:
            logger.warning("⚠️  No se obtuvieron partidos")
            return False
        
        logger.info(f"✅ {len(matches)} partidos obtenidos")
        
        # Análisis de partidos
        logger.info("🧠 Analizando partidos...")
        analyzed_matches = []
        analysis_errors = 0
        
        for i, match in enumerate(matches):
            try:
                logger.info(f"Analizando {i+1}/{len(matches)}: {match['home_team']} vs {match['away_team']}")
                analysis = analyzer.analyze_match(match)
                match['analysis'] = analysis
                analyzed_matches.append(match)
                
            except Exception as e:
                analysis_errors += 1
                logger.error(f"❌ Error analizando partido {match.get('id', 'N/A')}: {e}")
                # Continúar con el siguiente partido
                continue
        
        # Actualizar cache
        CACHE['scraped_data'] = matches
        CACHE['analysis_results'] = analyzed_matches
        CACHE['last_update'] = datetime.now()
        CACHE['error_count'] = analysis_errors
        
        update_duration = time.time() - update_start_time
        
        logger.info(f"✅ Actualización completada en {update_duration:.2f}s")
        logger.info(f"📊 Partidos analizados: {len(analyzed_matches)}/{len(matches)}")
        if analysis_errors > 0:
            logger.warning(f"⚠️  Errores de análisis: {analysis_errors}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error crítico en actualización: {e}")
        CACHE['error_count'] += 1
        return False

def should_update_data():
    """Verifica si se debe actualizar los datos"""
    if not CACHE.get('last_update'):
        return True
    
    time_diff = datetime.now() - CACHE['last_update']
    return time_diff.total_seconds() > CACHE['update_frequency']

# ============================================================
# ENDPOINTS DE LA API REST
# ============================================================

@app.route('/')
def dashboard():
    """Dashboard principal mejorado"""
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        matches_count = len(CACHE.get('analysis_results', []))
        scraped_count = len(CACHE.get('scraped_data', []))
        
        status = "✅ OPERATIVO" if CACHE.get('initialization_complete') else "🔄 INICIALIZANDO"
        
        last_update = CACHE.get('last_update')
        if last_update:
            last_update_str = last_update.strftime('%Y-%m-%d %H:%M:%S')
            time_since_update = datetime.now() - last_update
            hours_since = int(time_since_update.total_seconds() / 3600)
        else:
            last_update_str = "Nunca"
            hours_since = 999
        
        sklearn_status = "✅ Activo" if SKLEARN_AVAILABLE else "⚠️  No disponible"
        error_count = CACHE.get('error_count', 0)
        
        return f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🎯 BetPlay Colombia - Sistema de Análisis IA</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    background: linear-gradient(135deg, #0a0e27, #1a1f3a, #2a2f4a); 
                    color: white; 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    min-height: 100vh;
                    padding: 20px;
                }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ 
                    text-align: center; 
                    background: linear-gradient(135deg, #ff6b35, #f7931e, #ff8c42); 
                    padding: 40px; 
                    border-radius: 20px; 
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(255, 107, 53, 0.3);
                }}
                .header h1 {{ font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .stat-card {{ 
                    background: rgba(45,53,97,0.9); 
                    padding: 25px; 
                    border-radius: 15px; 
                    border-left: 5px solid #00ff88;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                    transition: transform 0.3s ease;
                }}
                .stat-card:hover {{ transform: translateY(-5px); }}
                .stat-value {{ font-size: 2.2em; color: #00ff88; margin: 15px 0; font-weight: bold; }}
                .stat-label {{ font-size: 1.1em; opacity: 0.8; }}
                .endpoint-section {{ 
                    background: rgba(45,53,97,0.8); 
                    padding: 30px; 
                    border-radius: 15px; 
                    margin: 20px 0; 
                    border-left: 6px solid #ff6b35;
                }}
                .endpoint {{ 
                    background: rgba(255,255,255,0.1); 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 8px; 
                    border-left: 3px solid #00ff88;
                    font-family: 'Courier New', monospace;
                }}
                .features-list {{ list-style: none; }}
                .features-list li {{ 
                    padding: 8px 0; 
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                    position: relative;
                    padding-left: 25px;
                }}
                .features-list li::before {{ 
                    content: '🚀'; 
                    position: absolute; 
                    left: 0;
                }}
                .status-indicator {{ 
                    display: inline-block; 
                    padding: 5px 15px; 
                    border-radius: 20px; 
                    background: rgba(0,255,136,0.2); 
                    border: 1px solid #00ff88;
                    margin: 5px 0;
                }}
                .warning {{ background: rgba(255,193,7,0.2); border-color: #ffc107; }}
                .error {{ background: rgba(220,53,69,0.2); border-color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 BetPlay Colombia</h1>
                    <p>Sistema Avanzado de Análisis Deportivo con Inteligencia Artificial</p>
                    <p>Version 2.1 - Optimizado para Render</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Estado del Sistema</div>
                        <div class="stat-value">{status}</div>
                        <div class="status-indicator {'warning' if error_count > 0 else ''}">
                            Errores: {error_count}
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Partidos Analizados</div>
                        <div class="stat-value">{matches_count}</div>
                        <div class="stat-label">Partidos obtenidos: {scraped_count}</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Última Actualización</div>
                        <div class="stat-value">{last_update_str}</div>
                        <div class="stat-label">Hace {hours_since} horas</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Machine Learning</div>
                        <div class="stat-value">{sklearn_status}</div>
                        <div class="stat-label">Sistema: {current_time}</div>
                    </div>
                </div>
                
                <div class="endpoint-section">
                    <h3>📡 API Endpoints Disponibles</h3>
                    <div class="endpoint">
                        <strong>GET /api/matches</strong><br>
                        Obtiene todos los partidos con análisis completo
                    </div>
                    <div class="endpoint">
                        <strong>GET /api/analysis</strong><br>
                        Resumen estadístico y mejores oportunidades
                    </div>
                    <div class="endpoint">
                        <strong>GET /api/recommendations</strong><br>
                        Recomendaciones de apuestas con mayor valor
                    </div>
                    <div class="endpoint">
                        <strong>POST /api/update</strong><br>
                        Fuerza actualización manual de datos
                    </div>
                    <div class="endpoint">
                        <strong>GET /health</strong><br>
                        Estado detallado del sistema
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
    except Exception as e:
        logger.error(f"Error en dashboard: {e}")
        return f"<h1>Error en Dashboard: {str(e)}</h1>", 500

@app.route('/api/matches')
def get_matches():
    """Endpoint para obtener todos los partidos analizados"""
    try:
        matches = CACHE.get('analysis_results', [])
        scraped_matches = CACHE.get('scraped_data', [])
        
        # Verificar si necesita actualización
        needs_update = should_update_data()
        
        response_data = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'total_matches': len(matches),
                'scraped_matches': len(scraped_matches),
                'last_update': CACHE.get('last_update').isoformat() if CACHE.get('last_update') else None,
                'needs_update': needs_update,
                'sklearn_available': SKLEARN_AVAILABLE,
                'error_count': CACHE.get('error_count', 0)
            },
            'matches': matches
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error en /api/matches: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
