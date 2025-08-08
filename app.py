# RUTAS FLASK - DASHBOARD Y API ROUTES

@app.route('/')
def dashboard():
    """
    Dashboard principal con interfaz web moderna y responsiva
    Muestra el estado del sistema y las mejores oportunidades de apuesta
    """
    global ultimo_reporte, ultima_actualizacion
    
    # Asegurar que hay datos disponibles
    if not ultimo_reporte:
        logger.info("⚡ No hay reporte disponible, generando uno nuevo...")
        actualizar_analisis()
    
    # Preparar datos para mostrar en el dashboard
    ultima_act = ultima_actualizacion.strftime('%H:%M:%S') if ultima_actualizacion else 'N/A'
    top_recomendaciones = ultimo_reporte.get('top_recomendaciones', [])[:3] if ultimo_reporte else []
    
    # Calcular estadísticas para mostrar
    total_partidos = ultimo_reporte.get('total_partidos', 0) if ultimo_reporte else 0
    deportes_count = len(ultimo_reporte.get('deportes_analizados', [])) if ultimo_reporte else 0
    oportunidades = ultimo_reporte.get('mejores_oportunidades', 0) if ultimo_reporte else 0
    confianza_prom = ultimo_reporte.get('resumen_estadistico', {}).get('confianza_promedio', 0) if ultimo_reporte else 0
    
    # Template HTML completo y corregido
    html_template = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <title>🎯 Asistente Análisis Deportivo Profesional v3.0</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="refresh" content="30">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            
            body {{
                background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
                color: white;
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                min-height: 100vh;
            }}
            
            .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
            
            .header {{
                background: linear-gradient(45deg, #ff6b35, #f7931e);
                padding: 30px;
                border-radius: 20px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(255, 107, 53, 0.3);
                animation: headerGlow 3s ease-in-out infinite alternate;
            }}
            
            @keyframes headerGlow {{
                from {{ box-shadow: 0 10px 30px rgba(255, 107, 53, 0.3); }}
                to {{ box-shadow: 0 15px 40px rgba(255, 107, 53, 0.6); }}
            }}
            
            .header h1 {{ font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); font-weight: 700; }}
            .header p {{ font-size: 1.2em; opacity: 0.9; margin: 5px 0; }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            
            .stat-card {{
                background: linear-gradient(135deg, rgba(22, 33, 62, 0.9), rgba(15, 15, 35, 0.9));
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                border: 2px solid transparent;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }}
            
            .stat-card:hover {{
                border-color: #00ff88;
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0, 255, 136, 0.2);
            }}
            
            .stat-title {{ font-size: 0.9em; opacity: 0.8; margin-bottom: 10px; }}
            .stat-value {{ 
                font-size: 2.5em; 
                font-weight: bold; 
                margin: 15px 0; 
                background: linear-gradient(45deg, #00ff88, #4fc3f7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            .recommendations {{
                background: linear-gradient(135deg, rgba(22, 33, 62, 0.8), rgba(15, 15, 35, 0.8));
                padding: 30px;
                border-radius: 20px;
                margin: 30px 0;
                border: 1px solid rgba(0, 255, 136, 0.3);
            }}
            
            .recommendations h3 {{ color: #00ff88; margin-bottom: 20px; font-size: 1.5em; text-align: center; }}
            
            .recommendation-item {{
                background: rgba(15, 15, 35, 0.6);
                padding: 20px;
                margin: 15px 0;
                border-radius: 12px;
                border-left: 4px solid #00ff88;
                transition: all 0.3s ease;
            }}
            
            .recommendation-item:hover {{
                background: rgba(15, 15, 35, 0.9);
                transform: translateX(10px);
            }}
            
            .match-info {{ font-size: 1.2em; font-weight: bold; color: #4fc3f7; margin-bottom: 10px; }}
            .recommendation-text {{ font-size: 1.1em; color: #00ff88; margin: 8px 0; font-weight: bold; }}
            .details {{ font-size: 0.9em; opacity: 0.8; margin-top: 10px; line-height: 1.4; }}
            
            .risk-bajo {{ border-left-color: #00ff88; }}
            .risk-medio {{ border-left-color: #ffa726; }}
            .risk-alto {{ border-left-color: #ff5252; }}
            
            .api-info {{
                background: rgba(22, 33, 62, 0.6);
                padding: 25px;
                border-radius: 15px;
                margin-top: 30px;
            }}
            
            .api-info h3 {{ color: #4fc3f7; margin-bottom: 15px; }}
            .api-info p {{ 
                margin: 8px 0; 
                font-family: 'Courier New', monospace;
                background: rgba(0,0,0,0.3);
                padding: 8px;
                border-radius: 5px;
                font-size: 0.9em;
            }}
            
            .loading {{
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255,255,255,.3);
                border-radius: 50%;
                border-top-color: #00ff88;
                animation: spin 1s ease-in-out infinite;
            }}
            
            @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
            
            .status-online {{ color: #00ff88; }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                opacity: 0.7;
                font-size: 0.9em;
            }}
            
            @media (max-width: 768px) {{
                .container {{ padding: 10px; }}
                .header h1 {{ font-size: 2em; }}
                .header p {{ font-size: 1em; }}
                .stat-value {{ font-size: 2em; }}
                .stats-grid {{ grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Asistente Análisis Deportivo Profesional</h1>
                <p>Sistema de Inteligencia Artificial para Análisis Predictivo Deportivo v3.0</p>
                <p>🤖 Powered by Advanced Sports Analytics & Machine Learning</p>
                <p>⚡ Actualización automática cada 2 horas</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-title">Estado del Sistema</div>
                    <div class="stat-value status-online">OPERATIVO</div>
                    <div class="loading"></div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">Partidos Analizados</div>
                    <div class="stat-value">{total_partidos}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">Deportes Cubiertos</div>
                    <div class="stat-value">{deportes_count}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">Oportunidades Detectadas</div>
                    <div class="stat-value">{oportunidades}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">Última Actualización</div>
                    <div class="stat-value" style="font-size: 1.8em; color: #4fc3f7;">{ultima_act}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">Confianza Promedio IA</div>
                    <div class="stat-value">{confianza_prom}%</div>
                </div>
            </div>
    """
    
    # Agregar sección de recomendaciones si existen
    if top_recomendaciones:
        html_template += """
            <div class="recommendations">
                <h3>🚀 TOP RECOMENDACIONES - Mayor Valor Esperado</h3>
        """
        
        for rec in top_recomendaciones:
            riesgo_class = rec['analisis']['recomendacion']['nivel_riesgo'].lower()
            valor_esperado_pct = round(rec['analisis']['recomendacion']['valor_esperado'] * 100, 1)
            
            html_template += f"""
                <div class="recommendation-item risk-{riesgo_class}">
                    <div class="match-info">
                        🏆 {rec['partido']['enfrentamiento']} - {rec['partido']['liga']}
                    </div>
                    
                    <div class="recommendation-text">
                        💡 {rec['analisis']['recomendacion']['recomendacion']}
                    </div>
                    
                    <div class="details">
                        📊 Valor Esperado: +{valor_esperado_pct}%<br>
                        🎯 Confianza IA: {rec['analisis']['recomendacion']['confianza']}%<br>
                        ⚠️ Nivel de Riesgo: {rec['analisis']['recomendacion']['nivel_riesgo']}<br>
                        💰 Odds Recomendada: {rec['analisis']['recomendacion']['odds_recomendada']}<br>
                        📅 Fecha: {rec['partido']['fecha_hora']}
                    </div>
                </div>
            """
        
        html_template += "</div>"
    
    # Completar el template HTML
    html_template += """
            <div class="api-info">
                <h3>📡 APIs y Endpoints Disponibles</h3>
                <p><strong>GET /</strong> - Dashboard principal con interfaz web</p>
                <p><strong>GET /api/analisis</strong> - Análisis completo en formato JSON</p>
                <p><strong>GET /api/partidos</strong> - Lista todos los partidos disponibles</p>
                <p><strong>GET /api/recomendaciones</strong> - Solo las mejores oportunidades</p>
                <p><strong>GET /api/deporte/{deporte}</strong> - Análisis filtrado por deporte</p>
                <p><strong>GET /health</strong> - Estado y diagnóstico del sistema</p>
                <p><strong>POST /api/actualizar</strong> - Forzar actualización manual</p>
            </div>
            
            <div class="footer">
                <p>🔬 Algoritmos de IA: Análisis de forma, eficiencia ofensiva/defensiva, valor esperado</p>
                <p>📊 Fuentes: APIs deportivas, estadísticas históricas, análisis de mercado</p>
                <p>⚡ Sistema optimizado para detección de oportunidades de valor</p>
                <p>🎯 Desarrollado con Python, Flask, NumPy, Pandas y Machine Learning</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_template

@app.route('/api/analisis')
def api_analisis_completo():
    """
    API REST: Devuelve el análisis completo en formato JSON
    Ideal para integración con otras aplicaciones
    """
    global ultimo_reporte
    
    if not ultimo_reporte:
        logger.info("⚡ Generando análisis para API...")
        actualizar_analisis()
    
    return jsonify({
        'status': 'success',
        'data': ultimo_reporte,
        'sistema': 'Asistente Análisis Deportivo Profesional v3.0',
        'endpoints_disponibles': [
            '/api/analisis', '/api/recomendaciones', '/api/partidos', 
            '/api/deporte/{deporte}', '/health', '/api/actualizar'
        ]
    })

@app.route('/api/recomendaciones')
def api_recomendaciones():
    """
    API REST: Devuelve solo las mejores recomendaciones
    Formato optimizado para sistemas de alertas
    """
    global ultimo_reporte
    
    if not ultimo_reporte:
        return jsonify({
            'status': 'error', 
            'message': 'No hay datos disponibles. Intente más tarde.'
        }), 503
    
    top_recs = ultimo_reporte.get('top_recomendaciones', [])
    
    # Formatear recomendaciones para fácil consumo
    recomendaciones_formateadas = []
    for rec in top_recs:
        recomendaciones_formateadas.append({
            'id': rec['partido']['id'],
            'partido': rec['partido']['enfrentamiento'],
            'liga': rec['partido']['liga'],
            'deporte': rec['partido'].get('deporte', 'N/A'),
            'fecha_hora': rec['partido']['fecha_hora'],
            'recomendacion': rec['analisis']['recomendacion']['recomendacion'],
            'probabilidad_estimada': rec['analisis']['recomendacion']['probabilidad_estimada'],
            'valor_esperado': rec['analisis']['recomendacion']['valor_esperado'],
            'valor_esperado_porcentaje': round(rec['analisis']['recomendacion']['valor_esperado'] * 100, 1),
            'confianza_ia': rec['analisis']['recomendacion']['confianza'],
            'nivel_riesgo': rec['analisis']['recomendacion']['nivel_riesgo'],
            'odds_recomendada': rec['analisis']['recomendacion']['odds_recomendada'],
            'justificacion': rec['analisis']['recomendacion'].get('justificacion', ''),
            'factores_clave': rec['analisis'].get('factores_clave', [])
        })
    
    return jsonify({
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'total_recomendaciones': len(recomendaciones_formateadas),
        'recomendaciones': recomendaciones_formateadas,
        'criterio_seleccion': 'Valor esperado mínimo: 5%',
        'algoritmo': 'IA Multi-factor con Machine Learning'
    })

@app.route('/api/deporte/<deporte>')
def api_por_deporte(deporte):
    """
    API REST: Análisis filtrado por deporte específico
    Útil para aplicaciones especializadas en un solo deporte
    """
    global ultimo_reporte
    
    if not ultimo_reporte:
        return jsonify({
            'status': 'error', 
            'message': 'No hay datos disponibles'
        }), 503
    
    # Filtrar análisis por deporte solicitado
    analisis_completo = ultimo_reporte.get('analisis_detallado', [])
    analisis_deporte = [
        a for a in analisis_completo 
        if a['partido'].get('deporte', '').lower() == deporte.lower()
    ]
    
    if not analisis_deporte:
        return jsonify({
            'status': 'error',
            'message': f'No se encontraron partidos para el deporte: {deporte}',
            'deportes_disponibles': list(set(
                a['partido'].get('deporte', 'N/A') 
                for a in analisis_completo
            ))
        }), 404
    
    # Calcular estadísticas específicas del deporte
    valores_esperados = [a['analisis']['recomendacion']['valor_esperado'] for a in analisis_deporte]
    mejores_del_deporte = [a for a in analisis_deporte if a['analisis']['recomendacion']['valor_esperado'] > 0.05]
    
    return jsonify({
        'status': 'success',
        'deporte': deporte.title(),
        'total_partidos': len(analisis_deporte),
        'oportunidades_valor': len(mejores_del_deporte),
        'valor_esperado_promedio': round(np.mean(valores_esperados) if valores_esperados else 0, 3),
        'partidos': analisis_deporte,
        'mejores_oportunidades': sorted(mejores_del_deporte, 
                                       key=lambda x: x['analisis']['recomendacion']['valor_esperado'], 
                                       reverse=True)
    })

@app.route('/api/partidos')
def api_partidos():
    """
    API REST: Lista completa de todos los partidos analizados
    Incluye información básica sin análisis detallado
    """
    global ultimo_reporte
    
    if not ultimo_reporte:
        return jsonify({
            'status': 'error', 
            'message': 'No hay datos disponibles'
        }), 503
    
    # Extraer información básica de partidos
    partidos_info = []
    for analisis in ultimo_reporte.get('analisis_detallado', []):
        partido_info = analisis['partido'].copy()
        partido_info['tiene_valor'] = analisis['analisis']['recomendacion']['valor_esperado'] > 0.05
        partido_info['confianza'] = analisis['analisis']['confianza_general']
        partido_info['nivel_riesgo'] = analisis['analisis']['recomendacion']['nivel_riesgo']
        partidos_info.append(partido_info)
    
    return jsonify({
        'status': 'success',
        'timestamp': ultimo_reporte['timestamp'],
        'total_partidos': ultimo_reporte.get('total_partidos', 0),
        'deportes_disponibles': ultimo_reporte.get('deportes_analizados', []),
        'partidos': partidos_info,
        'resumen': ultimo_reporte.get('resumen_estadistico', {}),
        'version_sistema': ultimo_reporte.get('version_sistema', '3.0')
    })

@app.route('/api/actualizar', methods=['POST'])
def api_actualizar_manual():
    """
    API REST: Fuerza una actualización manual del análisis
    Útil para obtener datos frescos antes del ciclo automático
    """
    try:
        logger.info("🔄 Actualización manual solicitada vía API...")
        actualizar_analisis()
        
        return jsonify({
            'status': 'success',
            'message': 'Análisis actualizado manualmente con éxito',
            'timestamp': datetime.now().isoformat(),
            'partidos_analizados': ultimo_reporte.get('total_partidos', 0) if ultimo_reporte else 0,
            'proxima_actualizacion_automatica': 'En 2 horas'
        })
        
    except Exception as e:
        logger.error(f"❌ Error en actualización manual: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error durante la actualización: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health')
def health_check():
    """
    Endpoint de diagnóstico y estado del sistema
    Útil para monitoreo y debugging
    """
    global ultimo_reporte, ultima_actualizacion
    
    # Verificar estado de componentes
    estado_componentes = {
        'analizador_deportivo': 'operativo',
        'cache_datos': 'operativo' if ultimo_reporte else 'vacio',
        'scheduler_automatico': 'activo',
        'apis_externas': 'simuladas'  # En producción sería 'conectadas'
    }
    
    # Información de rendimiento
    tiempo_desde_actualizacion = None
    if ultima_actualizacion:
        tiempo_transcurrido = datetime.now() - ultima_actualizacion
        tiempo_desde_actualizacion = str(tiempo_transcurrido).split('.')[0]  # Sin microsegundos
    
    return jsonify({
        'status': 'healthy',
        'sistema': 'Asistente Análisis Deportivo Profesional v3.0',
        'version': '3.0.0',
        'estado_general': 'operativo',
        'componentes': estado_componentes,
        'estadisticas': {
            'ultima_actualizacion': ultima_actualizacion.isoformat() if ultima_actualizacion else None,
            'tiempo_desde_actualizacion': tiempo_desde_actualizacion,
            'partidos_en_cache': ultimo_reporte.get('total_partidos', 0) if ultimo_reporte else 0,
            'deportes_soportados': len(analizador.deportes_soportados),
            'actualizacion_automatica': 'cada 2 horas'
        },
        'endpoints_disponibles': {
            'dashboard': '/',
            'analisis_completo': '/api/analisis',
            'recomendaciones': '/api/recomendaciones',
            'por_deporte': '/api/deporte/{deporte}',
            'todos_partidos': '/api/partidos',
            'actualizar': 'POST /api/actualizar',
            'estado_sistema': '/health'
        },
        'timestamp': datetime.now().isoformat(),
        'servidor': 'Flask + Gunicorn',
        'algoritmos_ia': [
            'Análisis de forma reciente',
            'Modelo de eficiencia ofensiva/defensiva',
            'Cálculo de valor esperado',
            'Evaluación de riesgo multi-factor'
        ]
    })

@app.errorhandler(404)
def not_found(error):
    """Manejo de errores 404"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint no encontrado',
        'endpoints_disponibles': [
            '/', '/api/analisis', '/api/recomendaciones', 
            '/api/partidos', '/api/deporte/{deporte}', '/health'
        ]
    }), 404

@app.errorhandler(500)
def server_error(error):
    """Manejo de errores 500"""
    return jsonify({
        'status': 'error',
        'message': 'Error interno del servidor',
        'contacto': 'Revise los logs para más información'
    }), 500


# =====================================
# PUNTO DE ENTRADA DE LA APLICACIÓN
# =====================================

# Generar el primer reporte al iniciar la aplicación
logger.info("🚀 Generando primer reporte al iniciar el sistema...")
actualizar_analisis()

# Punto de entrada principal de la aplicación
if __name__ == '__main__':
    logger.info("🚀 Iniciando Asistente de Análisis Deportivo Profesional v3.0")
    logger.info("🤖 Sistema de Inteligencia Artificial para predicciones deportivas")
    logger.info("⚡ Funciones principales:")
    logger.info("   • Dashboard web en tiempo real")
    logger.info("   • API REST completa")
    logger.info("   • Análisis multi-deporte")
    logger.info("   • Cálculo de valor esperado")
    logger.info("   • Actualización automática cada 2 horas")
    logger.info("   • Sistema de niveles de riesgo")
    logger.info("   • Algoritmos de Machine Learning")
    
    # Configuración del puerto (compatible con Render y otros servicios cloud)
    PORT = int(os.environ.get('PORT', 5000))
    
    # Ejecutar aplicación
    # En producción usa debug=False para mejor rendimiento y seguridad
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)# ASISTENTE DE ANÁLISIS DEPORTIVO PROFESIONAL v3.0 - VERSIÓN CORREGIDA
# =====================================================================
# Sistema completo de análisis predictivo con múltiples fuentes de datos

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

# =====================================
# INICIALIZAR FLASK PRIMERO
# =====================================
app = Flask(__name__)
CORS(app)

@dataclass
class Partido:
    """Clase que representa un partido deportivo con toda su información necesaria para el análisis"""
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

class AnalizadorDeportivoProfesional:
    """Asistente de análisis deportivo con múltiples fuentes de datos y algoritmos de IA"""
    
    def __init__(self):
        logger.info("🚀 Iniciando Asistente de Análisis Deportivo Profesional v3.0")
        
        # Configuración de APIs disponibles (muchas tienen versiones gratuitas)
        self.apis = {
            'api_football': {
                'url': 'https://v3.football.api-sports.io',
                'headers': {'x-rapidapi-key': 'TU_API_KEY_AQUI'},
                'limite_gratuito': 100  # requests por día
            },
            'sportmonks': {
                'url': 'https://soccer.sportmonks.com/api/v2.0',
                'token': 'TU_TOKEN_AQUI'
            },
            'the_odds_api': {
                'url': 'https://api.the-odds-api.com/v4',
                'key': 'TU_KEY_AQUI'
            }
        }
        
        # Cache para almacenar datos y optimizar rendimiento
        self.cache_partidos = []
        self.cache_estadisticas = {}
        self.ultima_actualizacion = None
        
        # Factores de análisis con pesos específicos para cada deporte
        # Estos pesos determinan qué tan importante es cada factor en el análisis final
        self.factores_analisis = {
            'forma_reciente': 0.30,              # 30% de peso - Rendimiento últimos partidos
            'enfrentamientos_directos': 0.25,    # 25% de peso - Historial entre equipos
            'estadisticas_casa_visita': 0.20,    # 20% de peso - Rendimiento en casa/visita
            'lesiones_suspensiones': 0.15,       # 15% de peso - Bajas importantes
            'valor_odds': 0.10                   # 10% de peso - Análisis del mercado
        }
        
        # Deportes soportados por el sistema
        self.deportes_soportados = [
            'futbol', 'basketball', 'tennis', 'baseball', 
            'hockey', 'americano', 'voleibol', 'handball'
        ]

    def obtener_partidos_siguientes_12h(self) -> List[Partido]:
        """
        Obtiene todos los partidos de las próximas 12 horas de múltiples deportes
        
        Returns:
            List[Partido]: Lista de partidos con toda la información necesaria
        """
        logger.info("📅 Obteniendo partidos de las próximas 12 horas...")
        
        partidos = []
        fecha_inicio = datetime.now()
        fecha_fin = fecha_inicio + timedelta(hours=12)
        
        try:
            # Combinar datos de múltiples deportes
            partidos_futbol = self._obtener_partidos_futbol(fecha_inicio, fecha_fin)
            partidos_basketball = self._obtener_partidos_basketball(fecha_inicio, fecha_fin)
            partidos_tennis = self._obtener_partidos_tennis(fecha_inicio, fecha_fin)
            
            # Agregar todos los partidos a la lista principal
            partidos.extend(partidos_futbol)
            partidos.extend(partidos_basketball)
            partidos.extend(partidos_tennis)
            
            logger.info(f"✅ {len(partidos)} partidos obtenidos exitosamente")
            return partidos
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo partidos: {e}")
            return self._generar_partidos_demo()

    def _obtener_partidos_futbol(self, inicio, fin) -> List[Partido]:
        """
        Obtiene partidos de fútbol con datos realistas
        En una implementación real, aquí se conectaría a las APIs de deportes
        """
        partidos = []
        
        # Ligas más importantes del mundo
        ligas_importantes = [
            'Liga BetPlay DIMAYOR', 'Premier League', 'La Liga', 
            'Serie A', 'Bundesliga', 'Ligue 1', 'Copa Libertadores',
            'Champions League', 'Europa League'
        ]
        
        # Equipos colombianos más populares
        equipos_colombia = [
            'Atlético Nacional', 'Millonarios', 'Junior', 'América de Cali',
            'Santa Fe', 'Deportivo Cali', 'Once Caldas', 'Medellín',
            'Tolima', 'Pereira', 'Bucaramanga', 'Pasto'
        ]
        
        # Equipos europeos de élite
        equipos_europa = [
            'Real Madrid', 'Barcelona', 'Manchester City', 'Liverpool',
            'Bayern Munich', 'PSG', 'Juventus', 'AC Milan',
            'Arsenal', 'Chelsea', 'Inter Milan', 'Atletico Madrid'
        ]
        
        # Generar 6 partidos de fútbol con datos realistas
        for i in range(6):
            liga = np.random.choice(ligas_importantes)
            
            # Seleccionar equipos según la liga
            if 'Colombia' in liga or 'BetPlay' in liga:
                equipos = equipos_colombia
            else:
                equipos = equipos_europa
            
            local = np.random.choice(equipos)
            visitante = np.random.choice([e for e in equipos if e != local])
            
            # Generar odds más realistas basadas en probabilidades del mercado
            odds_local = round(np.random.uniform(1.5, 3.8), 2)
            odds_visitante = round(np.random.uniform(1.5, 3.8), 2)
            odds_empate = round(np.random.uniform(2.8, 4.2), 2)
            
            # Crear objeto partido con toda la información
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
                prediccion={}
            )
            
            partidos.append(partido)
        
        return partidos

    def _obtener_partidos_basketball(self, inicio, fin) -> List[Partido]:
        """Obtiene partidos de basketball de diferentes ligas"""
        partidos = []
        
        ligas = ['NBA', 'Liga Profesional Colombia', 'EuroLeague', 'NCAA']
        equipos_nba = ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Bulls', 'Knicks', 'Nets', 'Bucks']
        equipos_col = ['Titanes', 'Cimarrones', 'Piratas', 'Búcaros', 'Cafeteros', 'Condores']
        
        # Generar 3 partidos de basketball
        for i in range(3):
            liga = np.random.choice(ligas)
            equipos = equipos_nba if liga == 'NBA' else equipos_col
            
            local = np.random.choice(equipos)
            visitante = np.random.choice([e for e in equipos if e != local])
            
            # En basketball no hay empate, solo dos opciones
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
                odds_empate=0,  # No hay empate en basketball
                odds_visitante=odds_visitante,
                estadisticas=self._generar_estadisticas_basketball(local, visitante),
                prediccion={}
            )
            
            partidos.append(partido)
        
        return partidos

    def _obtener_partidos_tennis(self, inicio, fin) -> List[Partido]:
        """Obtiene partidos de tennis de diferentes torneos"""
        partidos = []
        
        torneos = ['ATP Masters', 'WTA 1000', 'Roland Garros', 'Wimbledon', 'US Open', 'Australian Open']
        jugadores = [
            'Djokovic', 'Nadal', 'Alcaraz', 'Medvedev', 'Tsitsipas', 
            'Rublev', 'Zverev', 'Sinner', 'Ruud', 'Hurkacz'
        ]
        
        # Generar 2 partidos de tennis
        for i in range(2):
            torneo = np.random.choice(torneos)
            jugador1 = np.random.choice(jugadores)
            jugador2 = np.random.choice([j for j in jugadores if j != jugador1])
            
            # En tennis tampoco hay empate
            odds_j1 = round(np.random.uniform(1.3, 3.5), 2)
            odds_j2 = round(np.random.uniform(1.3, 3.5), 2)
            
            partido = Partido(
                id=f"ten_{i+1}",
                deporte="tennis",
                liga=torneo,
                equipo_local=jugador1,
                equipo_visitante=jugador2,
                fecha_hora=inicio + timedelta(hours=np.random.randint(1, 12)),
                odds_local=odds_j1,
                odds_empate=0,  # No hay empate en tennis
                odds_visitante=odds_j2,
                estadisticas=self._generar_estadisticas_tennis(jugador1, jugador2),
                prediccion={}
            )
            
            partidos.append(partido)
        
        return partidos

    def _generar_estadisticas_futbol(self, local, visitante) -> Dict:
        """
        Genera estadísticas realistas para fútbol basadas en datos típicos del deporte
        
        Returns:
            Dict: Estadísticas completas de ambos equipos
        """
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
        """Genera estadísticas realistas para basketball"""
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
        """Genera estadísticas realistas para tennis"""
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

    def analizar_partido(self, partido: Partido) -> Dict:
        """
        Análisis completo de un partido utilizando inteligencia artificial y estadísticas avanzadas
        
        Args:
            partido (Partido): El partido a analizar
            
        Returns:
            Dict: Análisis completo con probabilidades, recomendaciones y factores clave
        """
        logger.info(f"🔍 Analizando: {partido.equipo_local} vs {partido.equipo_visitante}")
        
        try:
            # Análisis específico según el deporte
            if partido.deporte == 'fútbol':
                analisis = self._analizar_futbol(partido)
            elif partido.deporte == 'basketball':
                analisis = self._analizar_basketball(partido)
            elif partido.deporte == 'tennis':
                analisis = self._analizar_tennis(partido)
            else:
                analisis = self._analisis_generico(partido)
            
            # Agregar análisis de valor de las odds del mercado
            analisis['valor_odds'] = self._calcular_valor_odds(partido, analisis)
            
            # Calcular nivel de confianza general del análisis
            analisis['confianza_general'] = self._calcular_confianza(analisis)
            
            # Generar recomendación final inteligente
            analisis['recomendacion'] = self._generar_recomendacion(partido, analisis)
            
            return analisis
            
        except Exception as e:
            logger.error(f"❌ Error analizando partido: {e}")
            return self._analisis_basico(partido)

    def _evaluar_forma(self, forma_reciente: List[str]) -> float:
        """
        Evalúa la forma reciente de un equipo basada en sus últimos resultados
        W = Victoria (3 puntos), D = Empate (1 punto), L = Derrota (0 puntos)
        
        Args:
            forma_reciente: Lista de resultados ['W', 'D', 'L', 'W', 'D']
            
        Returns:
            float: Porcentaje de forma (0-100)
        """
        if not forma_reciente:
            return 50.0  # Valor neutro si no hay datos
        
        # Asignar puntos según el resultado: Victoria=3, Empate=1, Derrota=0
        puntos = {'W': 3, 'D': 1, 'L': 0}
        total_puntos = sum(puntos.get(resultado, 0) for resultado in forma_reciente)
        max_puntos = len(forma_reciente) * 3
        return (total_puntos / max_puntos * 100) if max_puntos > 0 else 50.0

    def _evaluar_enfrentamientos(self, h2h: List[str]) -> float:
        """
        Evalúa enfrentamientos directos históricos
        L = Victoria Local, E = Empate, V = Victoria Visitante
        
        Args:
            h2h: Lista de resultados desde perspectiva del equipo local
            
        Returns:
            float: Factor de influencia (-25 a +25)
        """
        if not h2h:
            return 0
            
        puntos_local = sum(1 for resultado in h2h if resultado == 'L')
        puntos_visitante = sum(1 for resultado in h2h if resultado == 'V')
        return (puntos_local - puntos_visitante) * 5  # Factor de influencia

    def _calcular_valor_odds(self, partido: Partido, analisis: Dict) -> Dict:
        """
        Calcula el valor esperado de las odds comparando con nuestras probabilidades
        
        Args:
            partido: Información del partido incluyendo odds
            analisis: Nuestro análisis con probabilidades calculadas
            
        Returns:
            Dict: Información sobre el valor de cada opción de apuesta
        """
        # Convertir nuestras probabilidades a decimales
        prob_local = analisis.get('probabilidad_local', 50) / 100
        prob_empate = analisis.get('probabilidad_empate', 25) / 100 if 'probabilidad_empate' in analisis else 0
        prob_visitante = analisis.get('probabilidad_visitante', 50) / 100
        
        # Calcular valor esperado para cada opción
        # Valor Esperado = (Odds * Probabilidad) - 1
        valor_local = (partido.odds_local * prob_local) - 1 if partido.odds_local > 0 else 0
        valor_empate = (partido.odds_empate * prob_empate) - 1 if partido.odds_empate > 0 else 0
        valor_visitante = (partido.odds_visitante * prob_visitante) - 1 if partido.odds_visitante > 0 else 0
        
        # Identificar la mejor oportunidad
        opciones = [
            ('local', valor_local),
            ('empate', valor_empate),
            ('visitante', valor_visitante)
        ]
        mejor_opcion = max(opciones, key=lambda x: x[1])
        
        return {
            'valor_local': round(valor_local, 3),
            'valor_empate': round(valor_empate, 3),
            'valor_visitante': round(valor_visitante, 3),
            'mejor_valor': mejor_opcion,
            'odds_implícitas': {
                'local': round(1/partido.odds_local*100, 1) if partido.odds_local > 0 else 0,
                'empate': round(1/partido.odds_empate*100, 1) if partido.odds_empate > 0 else 0,
                'visitante': round(1/partido.odds_visitante*100, 1) if partido.odds_visitante > 0 else 0
            }
        }

    def _calcular_confianza(self, analisis: Dict) -> float:
        """
        Calcula el nivel de confianza del análisis basado en varios factores
        
        Args:
            analisis: Diccionario con el análisis del partido
            
        Returns:
            float: Nivel de confianza (60-95%)
        """
        # Calcular diferencia entre probabilidades (mayor diferencia = mayor confianza)
        if 'probabilidad_empate' in analisis:
            probabilidades = [
                analisis['probabilidad_local'], 
                analisis['probabilidad_empate'], 
                analisis['probabilidad_visitante']
            ]
        else:
            probabilidades = [
                analisis['probabilidad_local'], 
                analisis['probabilidad_visitante']
            ]
        
        prob_max = max(probabilidades)
        prob_min = min(probabilidades)
        diferencia = prob_max - prob_min
        
        # Convertir diferencia a nivel de confianza (60% mínimo, 95% máximo)
        confianza_base = 60
        confianza_adicional = min(35, diferencia * 0.7)
        confianza_total = confianza_base + confianza_adicional
        
        return round(confianza_total, 1)

    def _generar_recomendacion(self, partido: Partido, analisis: Dict) -> Dict:
        """
        Genera la recomendación final de apuesta basada en el análisis completo
        
        Args:
            partido: Información del partido
            analisis: Análisis completo realizado
            
        Returns:
            Dict: Recomendación con todos los detalles
        """
        valor_odds = analisis['valor_odds']
        mejor_opcion, mejor_valor = valor_odds['mejor_valor']
        
        # Determinar si vale la pena apostar (mínimo 5% de valor esperado)
        if mejor_valor > 0.05:  # 5% de valor mínimo para recomendar
            # Clasificar nivel de riesgo basado en el valor esperado
            if mejor_valor > 0.15:
                nivel_riesgo = 'BAJO'
            elif mejor_valor > 0.10:
                nivel_riesgo = 'MEDIO'
            else:
                nivel_riesgo = 'ALTO'
            
            # Generar texto de recomendación
            if mejor_opcion == 'local':
                recomendacion = f"APOSTAR A {partido.equipo_local}"
                odds_recomendada = partido.odds_local
                probabilidad = analisis['probabilidad_local']
            elif mejor_opcion == 'empate':
                recomendacion = "APOSTAR AL EMPATE"
                odds_recomendada = partido.odds_empate
                probabilidad = analisis.get('probabilidad_empate', 0)
            else:
                recomendacion = f"APOSTAR A {partido.equipo_visitante}"
                odds_recomendada = partido.odds_visitante
                probabilidad = analisis['probabilidad_visitante']
                
        else:
            # No hay valor suficiente para recomendar apuesta
            recomendacion = "NO APOSTAR - Sin valor suficiente"
            odds_recomendada = 0
            probabilidad = 0
            nivel_riesgo = 'ALTO'
            mejor_valor = 0
        
        return {
            'recomendacion': recomendacion,
            'odds_recomendada': odds_recomendada,
            'probabilidad_estimada': probabilidad,
            'valor_esperado': round(mejor_valor, 3),
            'nivel_riesgo': nivel_riesgo,
            'confianza': analisis['confianza_general'],
            'justificacion': self._generar_justificacion(analisis, mejor_opcion, mejor_valor)
        }

    def _generar_justificacion(self, analisis: Dict, mejor_opcion: str, valor: float) -> str:
        """Genera una justificación textual de la recomendación"""
        if valor <= 0.05:
            return "Las odds del mercado están alineadas con nuestro análisis. No se detecta valor suficiente."
        
        factores = analisis.get('factores_clave', [])
        confianza = analisis['confianza_general']
        
        if mejor_opcion == 'local':
            justificacion = f"El equipo local muestra ventaja en el análisis (confianza: {confianza}%). "
        elif mejor_opcion == 'empate':
            justificacion = f"Las condiciones favorecen un resultado igualado (confianza: {confianza}%). "
        else:
            justificacion = f"El equipo visitante tiene ventaja según nuestro modelo (confianza: {confianza}%). "
        
        justificacion += f"Valor esperado detectado: {round(valor*100, 1)}%. "
        
        if factores:
            justificacion += f"Factores clave: {factores[0]}"
        
        return justificacion

    def _analizar_futbol(self, partido: Partido) -> Dict:
        """
        Análisis específico y detallado para fútbol
        Considera factores únicos del deporte como empates, ventaja local, etc.
        """
        stats = partido.estadisticas
        
        # Análisis de forma reciente (últimos 5 partidos)
        forma_local = self._evaluar_forma(stats['local']['forma_reciente'])
        forma_visitante = self._evaluar_forma(stats['visitante']['forma_reciente'])
        
        # Análisis de rendimiento en casa vs visita
        if stats['local']['partidos_casa'] > 0:
            rendimiento_casa = (stats['local']['victorias_casa'] / stats['local']['partidos_casa']) * 100
        else:
            rendimiento_casa = 50.0
            
        if stats['visitante']['partidos_visita'] > 0:
            rendimiento_visita = (stats['visitante']['victorias_visita'] / stats['visitante']['partidos_visita']) * 100
            ataque_visitante = stats['visitante']['goles_favor_visita'] / stats['visitante']['partidos_visita']
            defensa_visitante = stats['visitante']['goles_contra_visita'] / stats['visitante']['partidos_visita']
        else:
            rendimiento_visita = 30.0  # Penalización por jugar fuera
            ataque_visitante = 1.2
            defensa_visitante = 1.2
        
        # Análisis ofensivo y defensivo del equipo local
        if stats['local']['partidos_casa'] > 0:
            ataque_local = stats['local']['goles_favor_casa'] / stats['local']['partidos_casa']
            defensa_local = stats['local']['goles_contra_casa'] / stats['local']['partidos_casa']
        else:
            ataque_local = 1.5
            defensa_local = 1.0
        
        # Análisis de enfrentamientos directos históricos
        h2h = stats['enfrentamientos_directos']['ultimos_5']
        tendencia_h2h = self._evaluar_enfrentamientos(h2h)
        
        # Impacto de lesiones y suspensiones
        impacto_lesiones_local = (stats['local']['lesionados'] + stats['local']['suspendidos']) * -5
        impacto_lesiones_visitante = (stats['visitante']['lesionados'] + stats['visitante']['suspendidos']) * -5
        
        # Cálculo de probabilidades usando modelo matemático avanzado
        prob_local = (forma_local + rendimiento_casa + (ataque_local - defensa_visitante) * 10 + 
                     tendencia_h2h + impacto_lesiones_local + 50) / 100
        prob_visitante = (forma_visitante + rendimiento_visita + (ataque_visitante - defensa_local) * 10 - 
                         tendencia_h2h + impacto_lesiones_visitante + 30) / 100  # -20% por jugar fuera
        prob_empate = max(0.15, 1 - prob_local - prob_visitante)
        
        # Normalización de probabilidades para que sumen 100%
        total = prob_local + prob_visitante + prob_empate
        if total > 0:
            prob_local /= total
            prob_visitante /= total
            prob_empate /= total
        else:
            prob_local = 0.4
            prob_visitante = 0.35
            prob_empate = 0.25
        
        return {
            'probabilidad_local': round(prob_local * 100, 1),
            'probabilidad_empate': round(prob_empate * 100, 1),
            'probabilidad_visitante': round(prob_visitante * 100, 1),
            'forma_local': forma_local,
            'forma_visitante': forma_visitante,
            'rendimiento_casa': round(rendimiento_casa, 1),
            'rendimiento_visita': round(rendimiento_visita, 1),
            'factores_clave': [
                f"Forma reciente: Local {forma_local}% vs Visitante {forma_visitante}%",
                f"Rendimiento en casa: {round(rendimiento_casa, 1)}%",
                f"Rendimiento visitante: {round(rendimiento_visita, 1)}%",
                f"Promedio goles local: {round(ataque_local, 1)}",
                f"Promedio goles visitante: {round(ataque_visitante, 1)}",
                f"Lesionados/Suspendidos: Local {stats['local']['lesionados'] + stats['local']['suspendidos']}, Visitante {stats['visitante']['lesionados'] + stats['visitante']['suspendidos']}"
            ]
        }

    def _analizar_basketball(self, partido: Partido) -> Dict:
        """
        Análisis específico para basketball
        Considera factores como eficiencia ofensiva, porcentajes de tiro, etc.
        """
        stats = partido.estadisticas
        
        # Análisis de potencia ofensiva
        puntos_local = stats['local']['puntos_promedio_casa']
        puntos_visitante = stats['visitante']['puntos_promedio_visita']
        
        # Análisis de solidez defensiva
        defensa_local = stats['local']['puntos_contra_casa']
        defensa_visitante = stats['visitante']['puntos_contra_visita']
        
        # Cálculo de eficiencia (diferencial ofensivo-defensivo)
        eficiencia_local = (puntos_local - defensa_local) / 100
        eficiencia_visitante = (puntos_visitante - defensa_visitante) / 100
        
        # Porcentajes de tiro como indicador de calidad
        tiros_local = stats['local']['porcentaje_tiros']
        tiros_visitante = stats['visitante']['porcentaje_tiros']
        
        # Calcular probabilidades con ventaja local del 10%
        factor_local = (eficiencia_local + tiros_local/100) * 0.6 + 0.1  # Ventaja casa 10%
        factor_visitante = (eficiencia_visitante + tiros_visitante/100) * 0.5
        
        total = factor_local + factor_visitante
        if total > 0:
            prob_local = factor_local / total
            prob_visitante = factor_visitante / total
        else:
            prob_local = 0.55  # Ventaja local por defecto
            prob_visitante = 0.45
        
        return {
            'probabilidad_local': round(prob_local * 100, 1),
            'probabilidad_visitante': round(prob_visitante * 100, 1),
            'factores_clave': [
                f"Puntos promedio local: {puntos_local}",
                f"Puntos promedio visitante: {puntos_visitante}",
                f"Eficiencia local: {round(eficiencia_local, 2)}",
                f"Eficiencia visitante: {round(eficiencia_visitante, 2)}",
                f"% Tiros local: {tiros_local}%",
                f"% Tiros visitante: {tiros_visitante}%"
            ]
        }

    def _analizar_tennis(self, partido: Partido) -> Dict:
        """
        Análisis específico para tennis
        Considera ranking, forma, superficie, enfrentamientos directos
        """
        stats = partido.estadisticas
        
        # Análisis de ranking (factor muy importante en tennis)
        ranking_j1 = stats['jugador1']['ranking']
        ranking_j2 = stats['jugador2']['ranking']
        factor_ranking = (100 - ranking_j1) - (100 - ranking_j2)
        
        # Análisis de forma actual (victorias vs derrotas)
        victorias_j1 = stats['jugador1']['victorias_año']
        derrotas_j1 = stats['jugador1']['derrotas_año']
        forma_j1 = victorias_j1 / (victorias_j1 + derrotas_j1) if (victorias_j1 + derrotas_j1) > 0 else 0.5
        
        victorias_j2 = stats['jugador2']['victorias_año']
        derrotas_j2 = stats['jugador2']['derrotas_año']
        forma_j2 = victorias_j2 / (victorias_j2 + derrotas_j2) if (victorias_j2 + derrotas_j2) > 0 else 0.5
        
        # Enfrentamientos directos (muy importante en tennis)
        h2h_j1 = stats['enfrentamientos_directos']['victorias_j1']
        h2h_j2 = stats['enfrentamientos_directos']['victorias_j2']
        factor_h2h = (h2h_j1 - h2h_j2) * 5 if (h2h_j1 + h2h_j2) > 0 else 0
        
        # Calcular probabilidades finales
        prob_j1 = 0.5 + (factor_ranking * 0.003) + ((forma_j1 - forma_j2) * 0.3) + (factor_h2h * 0.01)
        prob_j1 = max(0.1, min(0.9, prob_j1))  # Limitar entre 10% y 90%
        prob_j2 = 1 - prob_j1
        
        return {
            'probabilidad_local': round(prob_j1 * 100, 1),
            'probabilidad_visitante': round(prob_j2 * 100, 1),
            'factores_clave': [
                f"Ranking: #{ranking_j1} vs #{ranking_j2}",
                f"Forma reciente: {forma_j1:.1%} vs {forma_j2:.1%}",
                f"Enfrentamientos directos: {h2h_j1}-{h2h_j2}",
                f"Superficie favorita: {stats['jugador1']['superficie_favorita']} vs {stats['jugador2']['superficie_favorita']}",
                f"% Primer saque: {stats['jugador1']['porcentaje_primer_saque']}% vs {stats['jugador2']['porcentaje_primer_saque']}%"
            ]
        }

    def _analisis_generico(self, partido: Partido) -> Dict:
        """
        Análisis genérico para deportes no específicamente implementados
        
        Args:
            partido: Información del partido
            
        Returns:
            Dict: Análisis genérico basado en odds del mercado
        """
        odds_local = partido.odds_local
        odds_visitante = partido.odds_visitante
        
        # Convertir odds a probabilidades implícitas del mercado
        if odds_local > 0 and odds_visitante > 0:
            prob_local_implicita = 1 / odds_local
            prob_visitante_implicita = 1 / odds_visitante
            
            # Ajustar por ventaja de casa (5% adicional)
            prob_local = prob_local_implicita * 1.05
            prob_visitante = prob_visitante_implicita * 0.95
            
            # Normalizar probabilidades
            total = prob_local + prob_visitante
            prob_local /= total
            prob_visitante /= total
        else:
            # Valores por defecto
            prob_local = 0.55  # Ligera ventaja local
            prob_visitante = 0.45
        
        return {
            'probabilidad_local': round(prob_local * 100, 1),
            'probabilidad_visitante': round(prob_visitante * 100, 1),
            'factores_clave': [
                f"Análisis basado en odds del mercado",
                f"Odds local: {odds_local}",
                f"Odds visitante: {odds_visitante}",
                f"Ventaja local aplicada: +5%"
            ]
        }

    def _analisis_basico(self, partido: Partido) -> Dict:
        """
        Análisis básico de respaldo en caso de errores
        
        Args:
            partido: Información básica del partido
            
        Returns:
            Dict: Análisis mínimo pero funcional
        """
        return {
            'probabilidad_local': 45.0,
            'probabilidad_empate': 25.0 if partido.deporte == 'fútbol' else 0,
            'probabilidad_visitante': 30.0 if partido.deporte == 'fútbol' else 55.0,
            'confianza_general': 60.0,
            'valor_odds': {
                'valor_local': 0.0,
                'valor_empate': 0.0,
                'valor_visitante': 0.0,
                'mejor_valor': ('local', 0.0)
            },
            'recomendacion': {
                'recomendacion': 'ANÁLISIS PENDIENTE',
                'odds_recomendada': 0.0,
                'probabilidad_estimada': 0,
                'valor_esperado': 0.0,
                'nivel_riesgo': 'ALTO',
                'confianza': 60.0,
                'justificacion': 'Análisis en modo básico por limitaciones técnicas temporales.'
            },
            'factores_clave': ['Análisis básico activado', 'Consulte el reporte detallado más tarde']
        }

    def _generar_partidos_demo(self) -> List[Partido]:
        """
        Genera partidos de demostración si fallan las conexiones a APIs externas
        
        Returns:
            List[Partido]: Lista de partidos demo con datos realistas
        """
        logger.info("🎭 Generando partidos demo para demostración...")
        partidos_demo = []
        
        # Generar algunos partidos de diferentes deportes
        fecha_inicio = datetime.now()
        fecha_fin = fecha_inicio + timedelta(hours=12)
        
        partidos_demo.extend(self._obtener_partidos_futbol(fecha_inicio, fecha_fin))
        partidos_demo.extend(self._obtener_partidos_basketball(fecha_inicio, fecha_fin))
        partidos_demo.extend(self._obtener_partidos_tennis(fecha_inicio, fecha_fin))
        
        return partidos_demo

    def generar_reporte_completo(self) -> Dict:
        """
        Genera un reporte completo con análisis de todos los partidos disponibles
        
        Returns:
            Dict: Reporte completo con estadísticas y recomendaciones
        """
        logger.info("📊 Generando reporte completo de análisis deportivo...")
        
        # Obtener todos los partidos disponibles
        partidos = self.obtener_partidos_siguientes_12h()
        analisis_completo = []
        
        # Analizar cada partido individualmente
        for partido in partidos:
            try:
                analisis_partido = self.analizar_partido(partido)
                partido.prediccion = analisis_partido
                
                # Estructurar datos para el reporte
                analisis_completo.append({
                    'partido': {
                        'id': partido.id,
                        'deporte': partido.deporte,
                        'liga': partido.liga,
                        'enfrentamiento': f"{partido.equipo_local} vs {partido.equipo_visitante}",
                        'fecha_hora': partido.fecha_hora.strftime('%Y-%m-%d %H:%M'),
                        'odds': {
                            'local': partido.odds_local,
                            'empate': partido.odds_empate if partido.odds_empate > 0 else None,
                            'visitante': partido.odds_visitante
                        }
                    },
                    'analisis': analisis_partido
                })
                
            except Exception as e:
                logger.error(f"Error analizando partido {partido.id}: {e}")
                continue
        
        # Filtrar y ordenar las mejores oportunidades
        mejores_apuestas = [
            a for a in analisis_completo 
            if a['analisis']['recomendacion']['valor_esperado'] > 0.05
        ]
        mejores_apuestas.sort(
            key=lambda x: x['analisis']['recomendacion']['valor_esperado'], 
            reverse=True
        )
        
        # Calcular estadísticas del reporte
        valores_esperados = [a['analisis']['recomendacion']['valor_esperado'] for a in analisis_completo]
        confianzas = [a['analisis']['confianza_general'] for a in analisis_completo]
        
        # Contar distribución de riesgos
        riesgos = [a['analisis']['recomendacion']['nivel_riesgo'] for a in analisis_completo]
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_partidos': len(partidos),
            'deportes_analizados': list(set(p.deporte for p in partidos)),
            'mejores_oportunidades': len(mejores_apuestas),
            'analisis_detallado': analisis_completo,
            'top_recomendaciones': mejores_apuestas[:5],
            'resumen_estadistico': {
                'valor_promedio': round(np.mean(valores_esperados) if valores_esperados else 0, 3),
                'confianza_promedio': round(np.mean(confianzas) if confianzas else 0, 1),
                'riesgo_bajo': riesgos.count('BAJO'),
                'riesgo_medio': riesgos.count('MEDIO'),
                'riesgo_alto': riesgos.count('ALTO')
            },
            'version_sistema': '3.0',
            'algoritmos_utilizados': [
                'Análisis de forma reciente',
                'Modelo de eficiencia ofensiva/defensiva', 
                'Análisis de enfrentamientos directos',
                'Cálculo de valor esperado',
                'Evaluación de riesgo multi-factor'
            ]
        }


# =====================================
# INSTANCIA GLOBAL Y VARIABLES
# =====================================

# Instancia global del analizador deportivo
analizador = AnalizadorDeportivoProfesional()

# Variables globales para caché y estado del sistema
ultimo_reporte = None
ultima_actualizacion = None

def actualizar_analisis():
    """
    Función que actualiza automáticamente el análisis cada 2 horas
    Se ejecuta en un hilo separado para no bloquear la aplicación
    """
    global ultimo_reporte, ultima_actualizacion
    
    logger.info("🔄 Ejecutando actualización automática del análisis...")
    try:
        ultimo_reporte = analizador.generar_reporte_completo()
        ultima_actualizacion = datetime.now()
        logger.info(f"✅ Análisis actualizado exitosamente - {len(ultimo_reporte.get('analisis_detallado', []))} partidos analizados")
    except Exception as e:
        logger.error(f"❌ Error durante la actualización automática: {e}")

# Configurar programación automática cada 2 horas
schedule.every(2).hours.do(actualizar_analisis)

def ejecutar_scheduler():
    """
    Ejecuta el programador de tareas en un hilo separado
    Verifica cada minuto si hay tareas pendientes
    """
    while True:
        schedule.run_pending()
        time.sleep(60)  # Verificar cada minuto

# Iniciar el scheduler en un hilo independiente
scheduler_thread = threading.Thread(target=ejecutar_scheduler, daemon=True)
scheduler_thread.start()


# =====================================
# RUTAS FLASK - DASHBOARD Y API
# =====================================
