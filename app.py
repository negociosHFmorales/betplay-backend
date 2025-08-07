# BETPLAY COLOMBIA - BACKEND COMPLETO v3.0
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import sys
import random

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
PORT = int(os.environ.get('PORT', 10000))

# DATOS GLOBALES
logger.info("🚀 Iniciando BetPlay Colombia Backend v3.0")

def generar_partidos_completos():
    """Genera partidos realistas con mejor variedad"""
    logger.info("🎲 Generando partidos...")
    
    equipos_colombia = [
        'Atlético Nacional', 'Millonarios FC', 'Junior de Barranquilla', 'América de Cali',
        'Deportivo Cali', 'Santa Fe', 'Medellín', 'Bucaramanga', 'Tolima', 'Once Caldas'
    ]
    
    equipos_europa = [
        'Real Madrid', 'Barcelona', 'Manchester City', 'Bayern München',
        'Liverpool', 'Arsenal', 'Inter Milan', 'Borussia Dortmund',
        'Atletico Madrid', 'Juventus', 'Manchester United', 'Chelsea'
    ]
    
    ligas = {
        'colombia': ['Liga BetPlay DIMAYOR I', 'Liga BetPlay DIMAYOR II', 'Copa Colombia'],
        'europa': ['La Liga', 'Premier League', 'Bundesliga', 'Serie A', 'Champions League', 'Europa League']
    }
    
    partidos = []
    
    # Generar 12 partidos variados
    for i in range(12):
        if i % 3 == 0:  # Partidos colombianos
            home = random.choice(equipos_colombia)
            away = random.choice([e for e in equipos_colombia if e != home])
            liga = random.choice(ligas['colombia'])
            odds_base = [1.8, 3.2, 2.4]  # Odds más balanceados para Colombia
        else:  # Partidos europeos
            home = random.choice(equipos_europa)
            away = random.choice([e for e in equipos_europa if e != home])
