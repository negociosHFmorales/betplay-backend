import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Any
import schedule
import time
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import threading
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configurar logging para monitorear el funcionamiento del sistema
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# === CONFIGURACIÓN DE APIS EXTERNAS ===
BING_API_KEY = os.environ.get('BING_API_KEY', 'TU_API_KEY_AQUI')
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', 'TU_API_KEY_AQUI')

@dataclass
class Partido:
    id: str
    deporte: str
    liga: str
    equipo_local: str
    equipo_visitante: str
    fecha_hora: datetime
    odds_local: float
    odds_empate: float
    odds_visitante: float
    estadisticas: Dict
    prediccion: Dict = None
    ciudad: str = ""  # Nueva: para buscar clima

class AnalizadorDeportivoProfesional:
    def __init__(self):
        logger.info("🚀 Iniciando Asistente de Análisis Deportivo Profesional v3.0 con web/contexto")
        self.deportes_soportados = [
            'futbol', 'basketball', 'tennis', 'baseball', 
            'hockey', 'americano', 'voleibol', 'handball'
        ]
        self.cache_partidos = []
        self.cache_estadisticas = {}
        self.ultima_actualizacion = None

    def obtener_partidos_siguientes_12h(self) -> List[Partido]:
        logger.info("📅 Obteniendo partidos de las próximas 12 horas...")
        partidos = []
        fecha_inicio = datetime.now()
        fecha_fin = fecha_inicio + timedelta(hours=12)
        try:
            partidos_futbol = self._obtener_partidos_futbol(fecha_inicio, fecha_fin)
            partidos_basketball = self._obtener_partidos_basketball(fecha_inicio, fecha_fin)
            partidos_tennis = self._obtener_partidos_tennis(fecha_inicio, fecha_fin)
            partidos.extend(partidos_futbol)
            partidos.extend(partidos_basketball)
            partidos.extend(partidos_tennis)
            logger.info(f"✅ {len(partidos)} partidos obtenidos exitosamente")
            return partidos
        except Exception as e:
            logger.error(f"❌ Error obteniendo partidos: {e}")
            return self._generar_partidos_demo()

    def _obtener_partidos_futbol(self, inicio, fin) -> List[Partido]:
        partidos = []
        ligas_importantes = [
            'Liga BetPlay DIMAYOR', 'Premier League', 'La Liga', 
            'Serie A', 'Bundesliga', 'Ligue 1', 'Copa Libertadores',
            'Champions League', 'Europa League'
        ]
        equipos_colombia = [
            ('Atlético Nacional', 'Medellín'),
            ('Millonarios', 'Bogotá'),
            ('Junior', 'Barranquilla'),
            ('América de Cali', 'Cali'),
            ('Santa Fe', 'Bogotá'),
            ('Deportivo Cali', 'Cali'),
            ('Once Caldas', 'Manizales'),
            ('Medellín', 'Medellín'),
            ('Tolima', 'Ibagué'),
            ('Pereira', 'Pereira'),
            ('Bucaramanga', 'Bucaramanga'),
            ('Pasto', 'Pasto')
        ]
        equipos_europa = [
            ('Real Madrid', 'Madrid'), ('Barcelona', 'Barcelona'), ('Manchester City', 'Manchester'), 
            ('Liverpool', 'Liverpool'), ('Bayern Munich', 'Munich'), ('PSG', 'Paris'), 
            ('Juventus', 'Turin'), ('AC Milan', 'Milan'), ('Arsenal', 'London'), 
            ('Chelsea', 'London'), ('Inter Milan', 'Milan'), ('Atletico Madrid', 'Madrid')
        ]
        for i in range(6):
            liga = np.random.choice(ligas_importantes)
            if 'Colombia' in liga or 'BetPlay' in liga:
                equipos = equipos_colombia
            else:
                equipos = equipos_europa
            local, ciudad_local = equipos[np.random.randint(len(equipos))]
            visitante, ciudad_visitante = equipos[np.random.randint(len(equipos))]
            while visitante == local:
                visitante, ciudad_visitante = equipos[np.random.randint(len(equipos))]
            odds_local = round(np.random.uniform(1.5, 3.8), 2)
            odds_visitante = round(np.random.uniform(1.5, 3.8), 2)
            odds_empate = round(np.random.uniform(2.8, 4.2), 2)
            partido = Partido(
                id=f"fut_{i+1}",
                deporte="fútbol",
                liga=liga,
                equipo_local=local,
                equipo_visitante=visitante,
                fecha_hora=inicio + timedelta(hours=np.random.randint(1, 12)),
                odds_local=odds_local,
                odds_empate=odds_empate,
                odds_visitante=odds_visitante,
                estadisticas=self._generar_estadisticas_futbol(local, visitante),
                ciudad=ciudad_local
            )
            partidos.append(partido)
        return partidos

    def _obtener_partidos_basketball(self, inicio, fin) -> List[Partido]:
        partidos = []
        ligas = ['NBA', 'Liga Profesional Colombia', 'EuroLeague', 'NCAA']
        equipos_nba = [
            ('Lakers', 'Los Angeles'), ('Warriors', 'San Francisco'), ('Celtics', 'Boston'),
            ('Heat', 'Miami'), ('Bulls', 'Chicago'), ('Knicks', 'New York'), 
            ('Nets', 'Brooklyn'), ('Bucks', 'Milwaukee')
        ]
        equipos_col = [
            ('Titanes', 'Barranquilla'), ('Cimarrones', 'Chocó'), ('Piratas', 'Bogotá'), 
            ('Búcaros', 'Bucaramanga'), ('Cafeteros', 'Armenia'), ('Condores', 'Cundinamarca')
        ]
        for i in range(3):
            liga = np.random.choice(ligas)
            equipos = equipos_nba if liga == 'NBA' else equipos_col
            local, ciudad_local = equipos[np.random.randint(len(equipos))]
            visitante, ciudad_visitante = equipos[np.random.randint(len(equipos))]
            while visitante == local:
                visitante, ciudad_visitante = equipos[np.random.randint(len(equipos))]
            odds_local = round(np.random.uniform(1.4, 2.8), 2)
            odds_visitante = round(np.random.uniform(1.4, 2.8), 2)
            partido = Partido(
                id=f"bas_{i+1}",
                deporte="basketball",
                liga=liga,
                equipo_local=local,
                equipo_visitante=visitante,
                fecha_hora=inicio + timedelta(hours=np.random.randint(1, 12)),
                odds_local=odds_local,
                odds_empate=0,
                odds_visitante=odds_visitante,
                estadisticas=self._generar_estadisticas_basketball(local, visitante),
                ciudad=ciudad_local
            )
            partidos.append(partido)
        return partidos

    def _obtener_partidos_tennis(self, inicio, fin) -> List[Partido]:
        partidos = []
        torneos = ['ATP Masters', 'WTA 1000', 'Roland Garros', 'Wimbledon', 'US Open', 'Australian Open']
        jugadores = [
            'Djokovic', 'Nadal', 'Alcaraz', 'Medvedev', 'Tsitsipas', 
            'Rublev', 'Zverev', 'Sinner', 'Ruud', 'Hurkacz'
        ]
        ciudades_torneos = ['Paris', 'London', 'New York', 'Melbourne', 'Rome', 'Madrid']
        for i in range(2):
            torneo = np.random.choice(torneos)
            jugador1 = np.random.choice(jugadores)
            jugador2 = np.random.choice([j for j in jugadores if j != jugador1])
            odds_j1 = round(np.random.uniform(1.3, 3.5), 2)
            odds_j2 = round(np.random.uniform(1.3, 3.5), 2)
            ciudad = np.random.choice(ciudades_torneos)
            partido = Partido(
                id=f"ten_{i+1}",
                deporte="tennis",
                liga=torneo,
                equipo_local=jugador1,
                equipo_visitante=jugador2,
                fecha_hora=inicio + timedelta(hours=np.random.randint(1, 12)),
                odds_local=odds_j1,
                odds_empate=0,
                odds_visitante=odds_j2,
                estadisticas=self._generar_estadisticas_tennis(jugador1, jugador2),
                ciudad=ciudad
            )
            partidos.append(partido)
        return partidos

    def _generar_estadisticas_futbol(self, local, visitante) -> Dict:
        return {
            'local': {
                'forma_reciente': [np.random.choice(['W', 'D', 'L']) for _ in range(5)],
                'goles_favor_casa': np.random.randint(8, 25),
                'goles_contra_casa': np.random.randint(3, 18),
                'partidos_casa': np.random.randint(8, 15),
                'victorias_casa': np.random.randint(4, 12),
                'lesionados': np.random.randint(0, 4),
                'suspendidos': np.random.randint(0, 2),
                'posesion_promedio': round(np.random.uniform(42, 65), 1),
                'tiros_por_partido': round(np.random.uniform(8, 18), 1)
            },
            'visitante': {
                'forma_reciente': [np.random.choice(['W', 'D', 'L']) for _ in range(5)],
                'goles_favor_visita': np.random.randint(6, 22),
                'goles_contra_visita': np.random.randint(5, 20),
                'partidos_visita': np.random.randint(8, 15),
                'victorias_visita': np.random.randint(2, 10),
                'lesionados': np.random.randint(0, 4),
                'suspendidos': np.random.randint(0, 2),
                'posesion_promedio': round(np.random.uniform(35, 58), 1),
                'tiros_por_partido': round(np.random.uniform(6, 16), 1)
            },
            'enfrentamientos_directos': {
                'ultimos_5': [np.random.choice(['L', 'E', 'V']) for _ in range(5)],
                'goles_local_promedio': round(np.random.uniform(0.8, 2.5), 1),
                'goles_visitante_promedio': round(np.random.uniform(0.6, 2.2), 1),
                'total_partidos': np.random.randint(8, 20)
            }
        }

    def _generar_estadisticas_basketball(self, local, visitante) -> Dict:
        return {
            'local': {
                'puntos_promedio_casa': np.random.randint(105, 125),
                'puntos_contra_casa': np.random.randint(95, 120),
                'victorias_casa': np.random.randint(15, 25),
                'derrotas_casa': np.random.randint(5, 15),
                'porcentaje_tiros': round(np.random.uniform(42, 52), 1),
                'rebotes_promedio': round(np.random.uniform(40, 50), 1),
                'asistencias_promedio': round(np.random.uniform(20, 30), 1)
            },
            'visitante': {
                'puntos_promedio_visita': np.random.randint(100, 120),
                'puntos_contra_visita': np.random.randint(98, 125),
                'victorias_visita': np.random.randint(12, 22),
                'derrotas_visita': np.random.randint(8, 18),
                'porcentaje_tiros': round(np.random.uniform(40, 50), 1),
                'rebotes_promedio': round(np.random.uniform(38, 48), 1),
                'asistencias_promedio': round(np.random.uniform(18, 28), 1)
            }
        }

    def _generar_estadisticas_tennis(self, j1, j2) -> Dict:
        return {
            'jugador1': {
                'ranking': np.random.randint(1, 100),
                'victorias_año': np.random.randint(15, 45),
                'derrotas_año': np.random.randint(5, 20),
                'sets_ganados': np.random.randint(80, 150),
                'superficie_favorita': np.random.choice(['Clay', 'Hard', 'Grass']),
                'porcentaje_primer_saque': round(np.random.uniform(55, 75), 1),
                'puntos_ganados_saque': round(np.random.uniform(70, 85), 1)
            },
            'jugador2': {
                'ranking': np.random.randint(1, 100),
                'victorias_año': np.random.randint(15, 45),
                'derrotas_año': np.random.randint(5, 20),
                'sets_ganados': np.random.randint(80, 150),
                'superficie_favorita': np.random.choice(['Clay', 'Hard', 'Grass']),
                'porcentaje_primer_saque': round(np.random.uniform(55, 75), 1),
                'puntos_ganados_saque': round(np.random.uniform(70, 85), 1)
            },
            'enfrentamientos_directos': {
                'victorias_j1': np.random.randint(0, 8),
                'victorias_j2': np.random.randint(0, 8),
                'superficie_ultima_victoria_j1': np.random.choice(['Clay', 'Hard', 'Grass'])
            }
        }

    # --- NUEVO: FUNCIONES DE BÚSQUEDA WEB Y CLIMA ---
    def buscar_noticias_equipo(self, equipo, deporte):
        """Busca últimos titulares/noticias del equipo usando Bing."""
        if not BING_API_KEY or BING_API_KEY == "TU_API_KEY_AQUI":
            return ["API Bing no configurada"]
        url = "https://api.bing.microsoft.com/v7.0/news/search"
        headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
        params = {"q": f"{equipo} {deporte}", "count": 2, "mkt": "es-ES"}
        try:
            r = requests.get(url, headers=headers, params=params, timeout=5)
            noticias = r.json().get('value', [])
            return [f"{n['name']} ({n['url']})" for n in noticias]
        except Exception as e:
            logger.error(f"Error Bing Search: {e}")
            return ["Sin datos de noticias"]

    def obtener_clima_ciudad(self, ciudad):
        if not WEATHER_API_KEY or WEATHER_API_KEY == "TU_API_KEY_AQUI":
            return "API Clima no configurada"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={WEATHER_API_KEY}&lang=es&units=metric"
        try:
            r = requests.get(url, timeout=5)
            data = r.json()
            desc = data['weather'][0]['description']
            temp = data['main']['temp']
            return f"{desc}, {temp}°C"
        except Exception as e:
            logger.error(f"Error clima: {e}")
            return "Clima no disponible"

    def analizar_partido(self, partido: Partido) -> Dict:
        logger.info(f"🔍 Analizando: {partido.equipo_local} vs {partido.equipo_visitante}")
        try:
            if partido.deporte == 'fútbol':
                analisis = self._analizar_futbol(partido)
            elif partido.deporte == 'basketball':
                analisis = self._analizar_basketball(partido)
            elif partido.deporte == 'tennis':
                analisis = self._analizar_tennis(partido)
            else:
                analisis = self._analisis_generico(partido)
            analisis['valor_odds'] = self._calcular_valor_odds(partido, analisis)
            analisis['confianza_general'] = self._calcular_confianza(analisis)
            # --- Añadir contexto web ---
            noticias_local = self.buscar_noticias_equipo(partido.equipo_local, partido.deporte)
            noticias_visit = self.buscar_noticias_equipo(partido.equipo_visitante, partido.deporte)
            clima = self.obtener_clima_ciudad(partido.ciudad)
            analisis['noticias_local'] = noticias_local
            analisis['noticias_visitante'] = noticias_visit
            analisis['clima'] = clima
            analisis['recomendacion'] = self._generar_recomendacion(partido, analisis)
            analisis['reporte_final'] = self.resumen_final_recomendacion(partido, analisis, noticias_local + noticias_visit, clima)
            return analisis
        except Exception as e:
            logger.error(f"❌ Error analizando partido: {e}")
            return self._analisis_basico(partido)

    # ... Todas las funciones de análisis (_analizar_futbol, _analizar_basketball, etc) igual que en tu archivo original ...
    # ... (por espacio, no las repito aquí, pero puedes copiar y pegar el cuerpo de esas funciones tal como las tienes) ...

    # --- NUEVO: Resumen Final Mejorado ---
    def resumen_final_recomendacion(self, partido, analisis, noticias, clima):
        rec = analisis['recomendacion']
        texto = f"""
🏆 Partido: {partido.equipo_local} vs {partido.equipo_visitante} ({partido.liga})
📅 Fecha/Hora: {partido.fecha_hora.strftime('%Y-%m-%d %H:%M')}
🌦️ Clima esperado: {clima}
📰 Noticias relevantes: {'; '.join(noticias) if noticias else 'Sin noticias destacadas'}
🔍 Análisis IA: {rec['justificacion']}
➡️ RECOMENDACIÓN FINAL: {rec['recomendacion']}
Probabilidad estimada: {rec['probabilidad_estimada']}%
Valor esperado: {rec['valor_esperado']*100:.1f}%
Nivel de riesgo: {rec['nivel_riesgo']}
Confianza IA: {rec['confianza']}%
"""
        return texto

    # ... El resto de funciones de reporte, caché, actualizaciones, etc. igual que en tu archivo original ...

# ========== Instancia global, caché y programación como antes ==========
analizador = AnalizadorDeportivoProfesional()
ultimo_reporte = None
ultima_actualizacion = None

def actualizar_analisis():
    global ultimo_reporte, ultima_actualizacion
    logger.info("🔄 Ejecutando actualización automática del análisis...")
    try:
        ultimo_reporte = analizador.generar_reporte_completo()
        ultima_actualizacion = datetime.now()
        logger.info(f"✅ Análisis actualizado exitosamente - {len(ultimo_reporte.get('analisis_detallado', []))} partidos analizados")
    except Exception as e:
        logger.error(f"❌ Error durante la actualización automática: {e}")

schedule.every(2).hours.do(actualizar_analisis)
def ejecutar_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)  # Verificar cada minuto

scheduler_thread = threading.Thread(target=ejecutar_scheduler, daemon=True)
scheduler_thread.start()

# ========== Todas las rutas Flask y endpoints igual que tu archivo original ==========
# Puedes copiar el dashboard(), api_analisis_completo(), api_recomendaciones(), etc.
# El endpoint /api/analisis ahora incluye contexto de noticias y clima en cada partido.

# Punto de entrada:
if __name__ == '__main__':
    logger.info("🚀 Iniciando Asistente de Análisis Deportivo Profesional v3.0 (con contexto web)")
    PORT = int(os.environ.get('PORT', 5000))
    actualizar_analisis()
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
