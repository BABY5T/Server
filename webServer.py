from flask import Flask, jsonify, render_template
from flask_cors import CORS
import json

from dotenv import load_dotenv
import os 

# load .env
load_dotenv()

webPort = os.environ.get('webPort')

app = Flask(__name__)
CORS(app) # CORS를 전체 애플리케이션에 적용

app.jinja_env.variable_start_string = '[['
app.jinja_env.variable_end_string = ']]'

@app.route('/',methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/cry',methods=['GET'])
def data1():
    file_path='/Users/kim/IOTT/Server/output.json'
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

@app.route('/pose',methods=['GET'])
def data2():
    file_path='/Users/kim/IOTT/Server/result.json'
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    
if __name__ == '__main__':
    app.run(debug=True, port=webPort, host='0.0.0.0')
