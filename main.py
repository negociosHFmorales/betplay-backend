import os
import requests
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# === FUNCIÓN PARA OBTENER PARTIDOS REALES DE LA API ===
def partidos_reales():
    url = "https://free-api-live-football-data.p.rapidapi.com/football-get-all-matches-by-leagueid"
    querystring = {"leagueid": "42"}  # Premier League Inglaterra (puedes cambiar por otra liga)
    headers = {
        "X-RapidAPI-Key": "c520762708mshf386c8dfe0d3d57p107df4jsn5bf238e95d35",
        "X-RapidAPI-Host": "free-api-live-football-data.p.rapidapi.com"
    }
    try:
        resp = requests.get(url, headers=headers, params=querystring, timeout=10)
        data = resp.json()
        partidos = []
        for item in data.get("result", []):
            partidos.append({
                "deporte": "Fútbol",
                "liga": item.get("league_name", "Desconocida"),
                "local": item.get("event_home_team", "Local"),
                "visitante": item.get("event_away_team", "Visitante"),
                "fecha_hora": item.get("event_date", "Sin Fecha") + " " + item.get("event_time", ""),
                "ciudad": item.get("country_name", "N/A"),
                "analisis": {
                    "prob_local": "N/A",
                    "prob_empate": "N/A",
                    "prob_visitante": "N/A",
                    "recomendacion": "Datos reales: análisis no disponible en modo demo."
                }
            })
        return partidos
    except Exception as e:
        print(f"Error obteniendo datos reales: {e}")
        return []

# === DASHBOARD HTML ===
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
    <h1>🎯 Asistente de Análisis Deportivo (Datos Reales)</h1>
    <p>
      Mostrando partidos reales (Premier League) en vivo desde la API.<br>
      <span style="color:#00BFFF">¡Ya tienes datos reales en tu dashboard Flask + RapidAPI!</span>
    </p>
    <h2>Partidos Próximos</h2>
    <table>
      <tr>
        <th>Deporte</th>
        <th>Partido</th>
        <th>Fecha/Hora</th>
        <th>Liga</th>
        <th>Ciudad</th>
        <th>Recomendación</th>
      </tr>
      {% for p in partidos %}
      <tr>
        <td><span class="tag">{{p.deporte}}</span></td>
        <td>{{p.local}} <b>vs</b> {{p.visitante}}</td>
        <td>{{p.fecha_hora}}</td>
        <td>{{p.liga}}</td>
        <td>{{p.ciudad}}</td>
        <td class="reco">{{p.analisis.recomendacion}}</td>
      </tr>
      {% endfor %}
    </table>
    <div class="footer">
      API disponible:<br>
      <a href='/api/partidos'>/api/partidos</a>
      <br><br>
      © 2025 Asistente Deportivo Pro - Flask + RapidAPI
    </div>
  </div>
</body>
</html>
"""

# === FLASK ENDPOINTS ===

@app.route('/')
def dashboard():
    partidos = partidos_reales()
    return render_template_string(dashboard_html, partidos=partidos)

@app.route('/api/partidos')
def api_partidos():
    partidos = partidos_reales()
    return jsonify(partidos)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
