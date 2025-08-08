from flask import Flask, render_template_string, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def dashboard():
    # Aquí va el HTML bonito. Cambia este bloque por el dashboard real de tu asistente.
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Dashboard Apuestas Deportivas</title>
      <meta charset="UTF-8">
      <style>
        body { background: #181C2F; color: #fff; font-family: Arial, sans-serif; text-align:center;}
        .card { background:#23294D; border-radius:12px; margin:20px auto; max-width:600px; padding:30px;}
        h1 { color: #00FF99; }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>🎯 Asistente de Análisis Deportivo</h1>
        <p>Sistema funcionando. Pronto verás aquí el análisis de partidos, predicciones y recomendaciones.</p>
        <p>Endpoints API disponibles:</p>
        <ul>
          <li><a href='/api/analisis' style='color:#00FF99'>/api/analisis</a></li>
          <li><a href='/api/recomendaciones' style='color:#00FF99'>/api/recomendaciones</a></li>
          <li><a href='/api/partidos' style='color:#00FF99'>/api/partidos</a></li>
        </ul>
        <p>Si ves esto, tu backend funciona. Ahora puedes agregar el análisis real y el dashboard completo.</p>
      </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/api/analisis')
def analisis():
    # Aquí pondrás el análisis real, por ahora es un ejemplo
    return jsonify({
        "status": "success",
        "mensaje": "Aquí irá el análisis completo de partidos"
    })

@app.route('/api/recomendaciones')
def recomendaciones():
    return jsonify({
        "status": "success",
        "mensaje": "Aquí irán las mejores recomendaciones de apuesta"
    })

@app.route('/api/partidos')
def partidos():
    return jsonify({
        "status": "success",
        "mensaje": "Aquí irá la lista de partidos analizados"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
