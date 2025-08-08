from flask import Flask, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "API de Análisis Deportivo funcionando correctamente",
        "endpoints": [
            "/api/analisis",
            "/api/partidos",
            "/api/recomendaciones"
        ]
    })

@app.route('/api/test')
def test():
    return jsonify({
        "status": "success",
        "message": "Test endpoint working"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
