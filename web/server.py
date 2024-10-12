from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dags.cache import SQLiteCache
from dags.pipeline import StageStatus
from utils.convert import json2yaml
from dags.parser import load_pipelines_from_yaml

import os
import asyncio
import tempfile
import traceback


app = Flask(__name__)
# 设置 CORS，允许携带凭证
CORS(app, resources={r"/rerun_graph": {"origins": ["http://127.0.0.1:8000", "http://localhost:8000"]}}, supports_credentials=True)


# 新增一个函数来处理预检请求
def handle_options_request():
    response = make_response()
    origin = request.headers.get('Origin')
    if origin in ["http://127.0.0.1:8000", "http://localhost:8000"]:
        response.headers.add("Access-Control-Allow-Origin", origin)
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response

# 新增一个函数来处理错误
def handle_error(e):
    error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
    print(error_message)
    response = jsonify({"error": error_message})
    origin = request.headers.get('Origin')
    if origin in ["http://127.0.0.1:8000", "http://localhost:8000"]:
        response.headers.add("Access-Control-Allow-Origin", origin)
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response, 500

# 新增一个函数来处理图表的重新运行逻辑
def rerun_graph_logic(graph_json_str):
    graph_yaml, job_id = json2yaml(graph_json_str)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_file.write(graph_yaml)
        temp_file_path = temp_file.name
    
    print(f"YAML 内容已写入临时文件: {temp_file_path}")
    try:
        asyncio.run(load_pipelines_from_yaml(temp_file_path))
    except:
        pass
    response = jsonify({"message": "running", "data": {"job_id": job_id}})
    origin = request.headers.get('Origin')
    if origin in ["http://127.0.0.1:8000", "http://localhost:8000"]:
        response.headers.add("Access-Control-Allow-Origin", origin)
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response

@app.route('/rerun_graph', methods=['POST', 'OPTIONS'])
def rerun_graph():
    if request.method == "OPTIONS":
        return handle_options_request()
    elif request.method == "POST":
        try:
            graph_json_str = request.json.get('data')
            if not graph_json_str:
                return jsonify({"error": "缺少 data 参数"}), 400
            return rerun_graph_logic(graph_json_str)
        except Exception as e:
            return handle_error(e)
        
@app.route('/get_stage_status', methods=['GET'])
def get_stage_status():
    try:

        job_id = request.args.get('job_id')
        stage_name = request.args.get('stage_name')

        # 创建一个SQLiteCache实例
        cache = SQLiteCache()
        
        # 从SQLite数据库中读取stage状态
        status = asyncio.run(cache.read(f"{stage_name}"))
        
        if status is None:
            status = 'default'
        
        # 将状态转换为StageStatus枚举
        stage_status = StageStatus(status)
        
        # 返回状态
        response = jsonify({"status": stage_status.value})
        origin = request.headers.get('Origin')
        if origin in ["http://127.0.0.1:8000", "http://localhost:8000"]:
            response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response
    except Exception as e:
        return handle_error(e)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4242)
