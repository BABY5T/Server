from flask import Flask, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app) # CORS를 전체 애플리케이션에 적용

@app.route('/cry',methods=['GET'])
def data():
    file_path='/Users/kim/IOTT/output.json'
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    
if __name__ == '__main__':
    app.run(debug=True,port=3000,host='0.0.0.0')
