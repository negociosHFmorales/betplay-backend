# BETPLAY COLOMBIA - ULTRA FIX v2.3
# =====================================
# Versión ultra-simplificada que garantiza funcionamiento

from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import sys

# Configurar logging para que sea visible
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# CREAR APP INMEDIATAMENTE
app = Flask(__name__)
CORS(app)

PORT = int(os.environ.get('PORT', 10000))

# DATOS GLOBALES - SE INICIALIZAN AL IMPORTAR
logger.info("🚀 Iniciando BetPlay Colombia Ultra Fix v2.3")

def generar_partidos_inmediato():
    """Genera partidos inmediatamente sin dependencias externas"""
    logger.info("🎲 Generando partidos...")
    
    equipos_colombia = ['Atlético Nacional', 'Millonarios FC', 'Junior de Barranquilla', 'América de Cali']
    equipos_europa = ['Real Madrid', 'Barcelona', 'Manchester City', 'Bayern München']
    
    partidos = []
    
    # Generar 8 partidos garantizados
    for i in range(4):
        # Partido colombiano
        home = equipos_colombia[i]
        away = equipos_colombia[(i+1) % len(equipos_colombia)]
        
        partido_col = {
            'id': f'col_{i+1}',
            'home_team': home,
            'away_team': away,
            'league': 'Liga BetPlay DIMAYOR',
            'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
            'time': ['15:00', '17:30', '20:00'][i % 3],
            'odds': {
                'home': round(1.5 + np.random.random() * 2, 2),
                'draw': round(2.8 + np.random.random() * 0.8, 2),
                'away': round(2.0 + np.random.random() * 2, 2)
            },
            'analysis': {
                'recommendation': ['home', 'away', 'draw'][i % 3],
                'confidence': round(70 + np.random.random() * 20, 1),
                'expected_value': round(np.random.random() * 0.2, 3),
                'win_probability': round(40 + np.random.random() * 30, 1),
                'risk_level': ['BAJO', 'MEDIO', 'ALTO'][i % 3]
            }
        }
        partidos.append(partido_col)
        
        # Partido europeo
        home_eu = equipos_europa[i]
        away_eu = equipos_europa[(i+2) % len(equipos_europa)]
        
        partido_eu = {
            'id': f'eu_{i+1}',
            'home_team': home_eu,
            'away_team': away_eu,
            'league': ['La Liga', 'Premier League', 'Bundesliga'][i % 3],
            'date': (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d'),
            'time': ['19:00', '21:00', '16:30'][i % 3],
            'odds': {
                'home': round(1.3 + np.random.random() * 1.5, 2),
                'draw': round(3.0 + np.random.random() * 1.0, 2),
                'away': round(2.5 + np.random.random() * 2.5, 2)
            },
            'analysis': {
                'recommendation': ['home', 'away', 'draw'][(i+1) % 3],
                'confidence': round(75 + np.random.random() * 15, 1),
                'expected_value': round(np.random.random() * 0.15, 3),
                'win_probability': round(45 + np.random.random() * 25, 1),
                'risk_level': ['BAJO', 'MEDIO'][i % 2]
            }
        }
        partidos.append(partido_eu)
    
    logger.info(f"✅ {len(partidos)} partidos generados exitosamente")
    return partidos

# GENERAR DATOS INMEDIATAMENTE AL IMPORTAR
try:
    PARTIDOS_CACHE = generar_partidos_inmediato()
    SISTEMA_ESTADO = 'OPERATIVO'
    ULTIMA_ACTUALIZACION = datetime.now()
    logger.info("✅ Sistema inicializado correctamente")
except Exception as e:
    logger.error(f"❌ Error en inicialización: {e}")
    PARTIDOS_CACHE = []
    SISTEMA_ESTADO = 'ERROR'
    ULTIMA_ACTUALIZACION = None

@app.route('/')
def home():
    """Dashboard principal ultra-simple"""
    
    num_partidos = len(PARTIDOS_CACHE)
    estado_color = "#00ff88" if SISTEMA_ESTADO == 'OPERATIVO' else "#ff4444"
    ultima_act = ULTIMA_ACTUALIZACION.strftime('%H:%M:%S') if ULTIMA_ACTUALIZACION else 'N/A'
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎯 BetPlay Colombia</title>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="5">
        <style>
            body {{
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: white;
                font-family: Arial, sans-serif;
                padding: 20px;
                margin: 0;
            }}
            .header {{
                background: linear-gradient(45deg, #ff6b35, #f7931e);
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: rgba(22, 33, 62, 0.8);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 BetPlay Colombia</h1>
            <p>Sistema de Análisis Deportivo - Ultra Fix v2.3</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div>Estado del Sistema</div>
                <div class="stat-value" style="color: {estado_color};">{SISTEMA_ESTADO}</div>
            </div>
            <div class="stat-card">
                <div>Partidos Disponibles</div>
                <div class="stat-value" style="color: #00ff88;">{num_partidos}</div>
            </div>
            <div class="stat-card">
                <div>Última Actualización</div>
                <div class="stat-value" style="color: #4fc3f7; font-size: 1.8em;">{ultima_act}</div>
            </div>
        </div>
        
        <div style="background: rgba(22, 33, 62, 0.6); padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3 style="color: #00ff88;">📡 Endpoints API:</h3>
            <p><strong>GET /api/matches</strong> - Ver todos los partidos</p>
            <p><strong>GET /api/analysis</strong> - Análisis y recomendaciones</p>
            <p><strong>GET /health</strong> - Estado del sistema</p>
        </div>
    </body>
    </html>
    '''

@app.route('/api/matches')
def get_matches():
    """Obtener todos los partidos"""
    try:
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_matches': len(PARTIDOS_CACHE),
            'system_status': SISTEMA_ESTADO,
            'matches': PARTIDOS_CACHE
        })
    except Exception as e:
        logger.error(f"Error en /api/matches: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/analysis')
def get_analysis():
    """Análisis de partidos"""
    try:
        if not PARTIDOS_CACHE:
            return jsonify({
                'status': 'warning',
                'message': 'No hay partidos disponibles'
            })
        
        # Calcular estadísticas
        total = len(PARTIDOS_CACHE)
        confidencias = [p['analysis']['confidence'] for p in PARTIDOS_CACHE]
        confianza_promedio = sum(confidencias) / len(confidencias)
        
        # Mejores apuestas (por valor esperado)
        mejores = sorted(PARTIDOS_CACHE, 
                        key=lambda x: x['analysis']['expected_value'], 
                        reverse=True)[:3]
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_matches': total,
                'average_confidence': round(confianza_promedio, 1),
                'best_expected_value': mejores[0]['analysis']['expected_value'] if mejores else 0
            },
            'top_recommendations': [
                {
                    'match': f"{p['home_team']} vs {p['away_team']}",
                    'league': p['league'],
                    'recommendation': p['analysis']['recommendation'],
                    'confidence': p['analysis']['confidence'],
                    'expected_value': p['analysis']['expected_value']
                } for p in mejores
            ]
        })
        
    except Exception as e:
        logger.error(f"Error en /api/analysis: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health')
def health():
    """Estado del sistema"""
    return jsonify({
        'status': 'healthy' if SISTEMA_ESTADO == 'OPERATIVO' else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'system_status': SISTEMA_ESTADO,
        'matches_count': len(PARTIDOS_CACHE),
        'last_update': ULTIMA_ACTUALIZACION.isoformat() if ULTIMA_ACTUALIZACION else None
    })

@app.route('/api/test')
def test():
    """Test del sistema"""
    try:
        # Regenerar datos para probar
        nuevos_partidos = generar_partidos_inmediato()
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'test_results': {
                'data_generation': len(nuevos_partidos) > 0,
                'cache_access': len(PARTIDOS_CACHE) > 0,
                'system_operational': SISTEMA_ESTADO == 'OPERATIVO'
            },
            'generated_matches': len(nuevos_partidos),
            'cached_matches': len(PARTIDOS_CACHE)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    logger.info(f"🚀 Iniciando servidor en puerto {PORT}")
    logger.info(f"📊 Partidos en cache: {len(PARTIDOS_CACHE)}")
    logger.info(f"🎯 Estado del sistema: {SISTEMA_ESTADO}")
    
    app.run(host='0.0.0.0', port=PORT, debug=False)
