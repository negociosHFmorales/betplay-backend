import os
import random
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Simulación de partidos y análisis sencillo
def partidos_demo():
    ahora = datetime.now()
    partidos = [
        {
            "id": 1,
            "deporte": "Fútbol",
            "liga": "Liga BetPlay",
            "local": "Millonarios",
            "visitante": "Nacional",
            "fecha_hora": (ahora + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M'),
            "ciudad": "Bogotá"
        },
        {
            "id": 2,
            "deporte": "Basketball",
            "liga": "NBA",
            "local": "Lakers",
            "visitante": "Warriors",
            "fecha_hora": (ahora + timedelta(hours=5)).strftime('%Y-%m-%d %H:%M'),
            "ciudad": "Los Angeles"
        },
        {
            "id": 3,
            "deporte": "Tennis",
            "liga": "Wimbledon",
            "local": "Djokovic",
            "visitante": "Alcaraz",
            "fecha_hora": (ahora + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M'),
            "ciudad": "London"
        }
    ]
    return partidos

def analizar_partido(partido):
    # Análisis DEMO por deporte
    random.seed(partido["id"])
    if partido["deporte"] == "Fútbol":
        prob_local = round(random.uniform(0.35, 0.55), 2)
        prob_empate = round(random.uniform(0.2, 0.35), 2)
        prob_visitante = round(1 - prob_local - prob_empate, 2)
        recomendacion = (
            "Partido parejo. Se sugiere doble oportunidad local/empate." if prob_local > prob_visitante
            else "El visitante puede sorprender. Cuidado con el empate."
        )
    elif partido["deporte"] == "Basketball":
        prob_local = round(random.uniform(0.55, 0.70), 2)
        prob_empate = 0.00
        prob_visitante = round(1 - prob_local, 2)
        recomendacion = (
            "El local tiene ventaja clara por su ofensiva." if prob_local > 0.6
            else "Juego cerrado, mejor evitar apostar."
        )
    elif partido["deporte"] == "Tennis":
        prob_local = round(random.uniform(0.45, 0.65), 2)
        prob_empate = 0.00
        prob_visitante = round(1 - prob_local, 2)
        recomendacion = (
            f"Favorito: {partido['local']}" if prob_local > prob_visitante
            else f"Favorito: {partido['visitante']}"
        )
    else:
        prob_local = prob_empate = prob_visitante = 0.33
        recomendacion = "Sin datos suficientes."
    return {
        "prob_local": prob_local,
        "prob_empate": prob_empate,
        "prob_visitante": prob_visitante,
        "recomendacion": recomendacion
    }

# --------- DASHBOARD HTML ---------
dashboard_html = """
<!DOCTYPE html>
<html>
<head>
  <title>Dashboard Apuestas Deportivas</title>
  <meta charset="UTF-8">
  <style>
    body { background: #181C2F; color: #fff; font-family: Arial, sans-serif; margin:0;}
    .card { background:#23294D; border-radius:12px; margin:30px auto; max-width:700px; padding:35px 40px;}
    h1 { color: #00FF99; margin-bottom:6px;}
    h2 { color: #00BFFF; }
    table { width:100%; color:#fff; border-collapse: collapse; margin-top: 18px;}
    th, td { padding: 8px 2px; border-bottom: 1px solid #334;}
    .tag { background:#00FF99; color:#181C2F; border-radius:4px; padding:2px 8px; font-size:14px;}
    .reco { color:#FFD700; font-weight:bold; }
    .footer { color:#BBB; margin-top:25px; font-size:14px;}
    a { color:#00FF99; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🎯 Asistente de Análisis Deportivo</h1>
    <p>
      Análisis automático de partidos de fútbol, basketball y tennis.<br>
      <span style="color:#00BFFF">Predicciones y recomendaciones para apostar mejor.</span>
    </p>
    <h2>Partidos Próximos (DEMO)</h2>
    <table>
      <tr>
        <th>Deporte</th>
        <th>Partido</th>
        <th>Fecha/Hora</th>
        <th>Análisis</th>
        <th>Recomendación</th>
      </tr>
      {% for p in partidos %}
      <tr>
        <td><span class="tag">{{p.deporte}}</span></td>
        <td>
          {% if p.deporte == "Tennis" %}
            {{p.local}} vs {{p.visitante}}
          {% else %}
            {{p.local}} <b>vs</b> {{p.visitante}}<br>
            <small>{{p.liga}}</small>
          {% endif %}
        </td>
        <td>{{p.fecha_hora}}</td>
        <td>
          Local: {{p.analisis.prob_local*100}}%<br>
          {% if p.deporte == "Fútbol" %}Empate: {{p.analisis.prob_empate*100}}%<br>{% endif %}
          Visitante: {{p.analisis.prob_visitante*100}}%
        </td>
        <td class="reco">{{p.analisis.recomendacion}}</td>
      </tr>
      {% endfor %}
    </table>
    <div class="footer">
      API disponible:<br>
      <a href='/api/analisis'>/api/analisis</a> | 
      <a href='/api/recomendaciones'>/api/recomendaciones</a> | 
      <a href='/api/partidos'>/api/partidos</a>
      <br><br>
      © 2025 Asistente Deportivo Pro - DEMO Render
    </div>
  </div>
</body>
</html>
"""

# --------- FLASK ENDPOINTS ---------

@app.route('/')
def dashboard():
    partidos = partidos_demo()
    for p in partidos:
        p["analisis"] = analizar_partido(p)
    return render_template_string(dashboard_html, partidos=partidos)

@app.route('/api/partidos')
def api_partidos():
    partidos = partidos_demo()
    return jsonify(partidos)

@app.route('/api/analisis')
def api_analisis():
    partidos = partidos_demo()
    analisis = []
    for p in partidos:
        resultado = {"partido": f"{p['local']} vs {p['visitante']} ({p['deporte']})"}
        resultado.update(analizar_partido(p))
        analisis.append(resultado)
    return jsonify(analisis)

@app.route('/api/recomendaciones')
def api_recomendaciones():
    partidos = partidos_demo()
    recomendaciones = [
        {
            "partido": f"{p['local']} vs {p['visitante']} ({p['deporte']})",
            "recomendacion": analizar_partido(p)["recomendacion"]
        }
        for p in partidos
    ]
    return jsonify(recomendaciones)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
