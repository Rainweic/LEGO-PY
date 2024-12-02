# -*- coding: utf-8 -*
import json
from flask import Flask, Response, request, jsonify, make_response
from flask_cors import CORS
from dags.cache import SQLiteCache
from dags.pipeline import StageStatus
from utils.convert import json2yaml
from dags.parser import load_pipelines_from_yaml

import io
import os
import math
import psutil
import pickle
import logging
import asyncio
import traceback
import polars as pl
import cloudpickle
import ray
from user_count import UserCountActor
from output import load_output
from cleaner import Cleaner
app = Flask(__name__)
origins = ["http://127.0.0.1:8000", "http://localhost:8000", "http://lego-ui:8000", "http://10.222.107.184:8000"]

CORS(app, resources={r"/rerun_graph": {"origins": origins}}, supports_credentials=True)

# 初始化Ray
ray.init(ignore_reinit_error=True)

# 全局字典存储Ray引用
pipeline_tasks = {}

@ray.remote
class PipelineActor:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def run_pipeline(self, temp_file_path: str):
        try:
            return self.loop.run_until_complete(load_pipelines_from_yaml(temp_file_path))
        except Exception as e:
            logging.error(f"Pipeline执行出错: {str(e)}")
            logging.exception("完整错误栈:")
            raise e
            
    def __del__(self):
        try:
            self.loop.close()
        except Exception:
            pass


@app.route('/test')
def test():
    return {"message": "Hello from py-lego"}


# 新增一个函数来处理预检请求
def handle_options_request():
    response = make_response()
    origin = request.headers.get('Origin')
    if origin in origins:
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
    if origin in origins:
        response.headers.add("Access-Control-Allow-Origin", origin)
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response, 500


# 新增一个函数来处理图表的重新运行逻辑
async def run_graph_logic(graph_json_str, force_rerun):
    """处理图表运行逻辑，使用Ray管理任务"""
    graph_yaml, job_id = json2yaml(graph_json_str, force_rerun=force_rerun)
    config_dir = os.path.join('cache', job_id, 'config')
    os.makedirs(config_dir, exist_ok=True)
    temp_file_path = os.path.join(config_dir, 'graph_config.yaml')
    
    with open(temp_file_path, 'w') as temp_file:
        temp_file.write(graph_yaml)
    logging.info(f"YAML 内容已写入临时文件: {temp_file_path}")
    
    # 创建Ray actor并启动pipeline
    actor = PipelineActor.remote()
    task_ref = actor.run_pipeline.remote(temp_file_path)
    
    # 存储到全局字典
    pipeline_tasks[job_id] = {
        'actor': actor,
        'task': task_ref
    }
    
    response = jsonify({"message": "running", "data": {"job_id": job_id}})
    origin = request.headers.get('Origin')
    if origin in origins:
        response.headers.add("Access-Control-Allow-Origin", origin)
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response


@app.route('/run_graph', methods=['POST', 'OPTIONS'])
async def run_graph():
    if request.method == "OPTIONS":
        return handle_options_request()
    elif request.method == "POST":
        try:
            graph_json_str = request.json.get('data')
            if not graph_json_str:
                return jsonify({"error": "缺少 data 参数"}), 400
            return await run_graph_logic(graph_json_str, force_rerun=True)
        except Exception as e:
            return handle_error(e)
        

@app.route('/continue_graph', methods=['POST', 'OPTIONS'])
async def continue_graph():
    if request.method == "OPTIONS":
        return handle_options_request()
    elif request.method == "POST":
        try:
            graph_json_str = request.json.get('data')
            if not graph_json_str:
                return jsonify({"error": "缺少 data 参数"}), 400
            return await run_graph_logic(graph_json_str, force_rerun=False)
        except Exception as e:
            return handle_error(e)


@app.route('/get_stage_status', methods=['POST', 'OPTIONS'])
def get_stage_status():

    if request.method == "OPTIONS":
        return handle_options_request()
    elif request.method == "POST":
        try:

            data_list = request.json.get("data")

            # 创建一个SQLiteCache实例
            cache = SQLiteCache()
            
            ret = {}
            for node_info in data_list:
                name, job_id = node_info['id'], node_info['job_id']
                # 从SQLite数据库中读取stage状态
                status = asyncio.run(cache.read(f"{job_id}_{name}"))
            
                if status is None:
                    status = 'running'
            
                # 将状态转换为StageStatus枚举
                stage_status = StageStatus(status)

                ret[name] = stage_status.value

            logging.info(f"ret: {ret}")
                
            # 返回状态
            response = jsonify(ret)
            origin = request.headers.get('Origin')
            if origin in origins:
                response.headers.add("Access-Control-Allow-Origin", origin)
            response.headers.add("Access-Control-Allow-Credentials", "true")
            return response
        except Exception as e:
            return handle_error(e)


@app.route('/get_cpm_log')
async def get_cpm_log():
    job_id = request.args.get("job_id")
    stage_name = request.args.get("node_id")
    logging.info(f"job id: {job_id}, stage name: {stage_name}")

    try:
        # 构建日志文件路径
        base_path = os.path.join(os.path.dirname(__file__), "../cache", job_id, "logs")
        stage_log_path = os.path.join(base_path, f"{stage_name}.log")
        pipeline_log_path = os.path.join(base_path, "pipeline.log")

        # 读取日志内容
        stage_log_content = "日志不存在"
        pipeline_log_content = "日志不存在"
        if os.path.exists(stage_log_path):
            with open(stage_log_path, "r") as f:
                stage_log_content = f.read()
        if os.path.exists(pipeline_log_path):
            with open(pipeline_log_path, "r") as f:
                pipeline_log_content = f.read()

        # 构建响应
        response = jsonify({"stage_log": stage_log_content, "pipeline_log": pipeline_log_content})
        origin = request.headers.get('Origin')
        if origin in origins:
            response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response

    except Exception as e:
        return handle_error(e)
    

@app.route("/output")
async def get_output():
    n_rows_one_page = 100
    job_id = request.args.get("job_id")
    stage_name = request.args.get("node_id")
    page_idx = int(request.args.get("page_idx"))
    output_idx = int(request.args.get("output_idx"))
    logging.info(f"job id: {job_id}, stage name: {stage_name}, page idx: {page_idx}")
    print(f"job id: {job_id}, stage name: {stage_name}, page idx: {page_idx}")

    try:
        # 创建一个SQLiteCache实例
        cache = SQLiteCache()
        # 获取组件输出命名
        output_names = await cache.read(f"{job_id}_{stage_name}_output_names")
        if output_names:
            output_name = pickle.loads(output_names)[output_idx]
            logging.info(f"reading output: {output_name}")
            data_path = await cache.read(output_name)
            if data_path.endswith('.parquet'):
                # 使用 scan_parquet 来创建懒惰查询
                lazy_df = pl.scan_parquet(data_path)
                
                # 计算总行数
                total = lazy_df.select(pl.len()).collect().item()
                
                # 计算起始行和结束行
                start_row = page_idx * n_rows_one_page
                
                def handle_column(col, dtype):
                    """处理不同类型的列数据
                    
                    Args:
                        col: 列名
                        dtype: polars数据类型
                    """
                    print(dtype)
                    if dtype in [pl.Float32, pl.Float64, pl.Int16, pl.Int32, pl.Int64, pl.Int8]:
                        return pl.col(col).fill_nan(None)
                    elif dtype == pl.String:
                        return pl.col(col).fill_null("")
                    elif isinstance(dtype, pl.List):
                        # 处理列表类型：转换为字符串
                        return pl.col(col).map_elements(
                            lambda x: str(x.to_numpy().tolist()) if x is not None else "",
                            return_dtype=pl.String
                        ).alias(col)
                    elif isinstance(dtype, pl.Struct):
                        # 处理结构体类型：转换为JSON字符串
                        return pl.col(col).cast(pl.String).fill_null("").alias(col)
                    else:
                        # 其他类型尝试转为字符串
                        try:
                            return pl.col(col).cast(pl.String).fill_null("").alias(col)
                        except:
                            logging.warning(f"无法处理的数据类型 {dtype}，列 {col} 将被填充为空字符串")
                            return pl.lit("").alias(col)
                
                data = (
                    lazy_df
                        .slice(start_row, n_rows_one_page)
                        .select([
                            handle_column(col, dtype) for col, dtype in zip(lazy_df.columns, lazy_df.dtypes)
                        ])
                        .with_row_count("行号", offset=start_row + 1)
                        .collect()
                )
                
                cols = [{"title": col, "dataIndex": col, "key": col, "width": 75 if col == "行号" else 150} for col in data.columns]
                data_dict = data.to_dicts()
                response = jsonify({
                    "tableData": data_dict, 
                    "tableCols": cols, 
                    "totalRows": total, 
                    "pageSize": n_rows_one_page, 
                    "pageCount": math.ceil(total / n_rows_one_page)
                })
        else:
            response = jsonify({"data": "数据暂未生成"})
            logging.warning("读取数据输出失败")
        
        origin = request.headers.get('Origin')
        if origin in origins:
            response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response

    except Exception as e:
        return handle_error(e)


@app.route("/cols_schema")
async def get_schema():
    job_id = request.args.get("job_id")
    stage_name = request.args.get("node_id")
    output_idx = int(request.args.get("output_idx"))

    try:
        # 创建一个SQLiteCache实例
        cache = SQLiteCache()
        # 获取组件输出命名
        output_names = await cache.read(f"{job_id}_{stage_name}_output_names")
        if output_names:
            output_name = pickle.loads(output_names)[output_idx]
            logging.info(f"reading output: {output_name}")
            data_path = await cache.read(output_name)
            if data_path.endswith('.parquet'):
                # 使用 scan_parquet 来创建懒惰查询
                cols_dtype = {str(k): str(v) for k, v in pl.read_parquet_schema(data_path).items()}
                cols_schema = [{'key': name, 'name': name, 'type': dtype} for name, dtype in cols_dtype.items()]
                response = jsonify({"schema": cols_schema})
        else:
            response = jsonify({"data": "数据暂未生成"})
            logging.warning("读取数据输出失败")

        origin = request.headers.get('Origin')
        if origin in origins:
            response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response
    except Exception as e:
        return handle_error(e)


@app.route("/summary")
async def get_summary():
    job_id = request.args.get("job_id")
    stage_name = request.args.get("node_id")

    try:
        # 创建一个SQLiteCache实例
        cache = SQLiteCache()
        summary = await cache.read(f"{job_id}_{stage_name}_summary")
        # print(summary)
    except Exception as e:
        return handle_error(e)

    if summary:
        response = jsonify({"hasData": True, "datas": pickle.loads(summary)})
    else:
        response = jsonify({"hasData": False, "datas": []})

    # response = jsonify(bar_base())

    origin = request.headers.get('Origin')
    if origin in origins:
        response.headers.add("Access-Control-Allow-Origin", origin)
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response


@app.route('/list_files', methods=['GET'])
def list_files():
    try:
        # 获取请求参数中的路径，默认为根目录
        path = request.args.get('path', '/')
        
        # 确保路径安全，防止目录遍历攻击
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(os.path.abspath('/')):
            return jsonify({"error": "Invalid path"}), 403
            
        items = []
        if os.path.exists(abs_path):
            for item in os.listdir(abs_path):
                item_path = os.path.join(abs_path, item)
                try:
                    items.append({
                        "name": item,
                        "path": item_path,
                        "type": "directory" if os.path.isdir(item_path) else "file",
                        "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None,
                        "modified": os.path.getmtime(item_path)
                    })
                except Exception as e:
                    logging.error(f"获取文件信息失败: {str(e)}")
        
        response = jsonify({
            "current_path": abs_path,
            "items": items
        })
        
        origin = request.headers.get('Origin')
        if origin in origins:
            response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response
        
    except Exception as e:
        return handle_error(e)


@app.route('/terminate_pipeline', methods=['GET', 'OPTIONS'])
async def terminate_pipeline():
    """终止pipeline执行"""
    if request.method == "OPTIONS":
        return handle_options_request()
    elif request.method == "GET":
        try:
            job_id = request.args.get("job_id")
            if not job_id:
                return jsonify({"error": "Missing job_id parameter"}), 400
            
            # 从全局字典获取Ray引用
            task_info = pipeline_tasks.get(job_id)
            if not task_info:
                return jsonify({"error": "Task not found"}), 404
            
            try:
                # 从缓存中获取stage names
                cache = SQLiteCache()
                pipeline_obj = cloudpickle.loads(await cache.read(f"pipeline_{job_id}"))
                stage_names = pipeline_obj.pipeline.nodes
                
                # 终止actor和任务
                ray.kill(task_info['actor'])
                ray.cancel(task_info['task'])
                
                # 更新所有stage状态为FAILED
                for stage_name in stage_names:
                    await cache.write(f"{job_id}_{stage_name}", StageStatus.FAILED.value)
                    logging.info(f"已将stage {stage_name} 状态设置为FAILED")
                
                # 清理全局字典
                del pipeline_tasks[job_id]
                
                msg = f"Pipeline {job_id} 已终止，所有stage状态已设置为FAILED"
                logging.info(msg)
                
            except Exception as e:
                msg = f"终止Pipeline {job_id} 时出错: {str(e)}"
                logging.error(msg)
                return jsonify({"error": msg}), 500
            
            response = jsonify({"message": msg})
            origin = request.headers.get('Origin')
            if origin in origins:
                response.headers.add("Access-Control-Allow-Origin", origin)
            response.headers.add("Access-Control-Allow-Credentials", "true")
            return response
            
        except Exception as e:
            return handle_error(e)


@app.route('/ray_status')
def ray_status():
    """获取Ray集群状态信息"""
    if request.method == "OPTIONS":
        return handle_options_request()
    elif request.method == "GET":
        try:
            # 检查Ray是否已初始化
            if not ray.is_initialized():
                return jsonify({
                    "error": "Ray未初始化",
                    "status": "offline",
                    "visualization": {
                        "type": "gauge",
                        "data": [
                            {"name": "CPU使用率", "value": 0},
                            {"name": "内存使用率", "value": 0}
                        ]
                    }
                }), 200

            try:
                # 获取Ray集群状态
                actors = len(ray.state.actors())
                tasks = len(ray.state.tasks())
                ray_status = "online"
            except Exception as e:
                logging.error(f"获取Ray状态失败: {str(e)}")
                actors = 0
                tasks = 0
                ray_status = "error"
            
            # 获取系统资源使用情况
            try:
                cpu_percent = psutil.cpu_percent(interval=1)  # 添加1秒间隔以获得更准确的值
            except:
                cpu_percent = 0
                
            try:
                memory = psutil.virtual_memory()
                memory_used = round(memory.used / 1024 / 1024 / 1024, 2)  # GB
                memory_total = round(memory.total / 1024 / 1024 / 1024, 2)  # GB
                memory_percent = memory.percent
            except:
                memory_used = 0
                memory_total = 0
                memory_percent = 0
            
            status = {
                "ray_status": {
                    "status": ray_status,
                    "actors": actors,
                    "tasks": tasks
                },
                "system_status": {
                    "cpu_usage": f"{cpu_percent}%",
                    "memory_usage": f"{memory_used}GB/{memory_total}GB ({memory_percent}%)"
                },
                "visualization": {
                    "type": "gauge",
                    "data": [
                        {
                            "name": "CPU使用率",
                            "value": round(cpu_percent, 1),
                            "min": 0,
                            "max": 100,
                            "format": "{value}%"
                        },
                        {
                            "name": "内存使用率",
                            "value": round(memory_percent, 1),
                            "min": 0,
                            "max": 100,
                            "format": "{value}%"
                        }
                    ]
                }
            }
            
            logging.info(f"系统状态: {status}")  # 添加日志
            
            response = jsonify(status)
            origin = request.headers.get('Origin')
            if origin in origins:
                response.headers.add("Access-Control-Allow-Origin", origin)
            response.headers.add("Access-Control-Allow-Credentials", "true")
            return response
            
        except Exception as e:
            logging.error(f"获取系统状态失败: {str(e)}")  # 添加错误日志
            return handle_error(e)


# 添加心跳接口
@app.route('/user_heartbeat', methods=['POST', 'OPTIONS'])
def user_heartbeat():
    if request.method == "OPTIONS":
        return handle_options_request()
    
    try:
        user_id = request.json.get('user_id')
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        # 更新用户活跃状态
        count = ray.get(user_counter.heartbeat.remote(user_id))
        
        response = jsonify({
            "active_users": count
        })
        
        origin = request.headers.get('Origin')
        if origin in origins:
            response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add("Access-Control-Allow-Credentials", "true")
        print(response)
        return response

    except Exception as e:
        print(e)
        return handle_error(e)


# 添加获取用户数量接口
@app.route('/user_count', methods=['GET', 'OPTIONS'])
def get_user_count():
    if request.method == "OPTIONS":
        return handle_options_request()
    
    try:
        count = ray.get(user_counter.get_count.remote())
        
        response = jsonify({
            "active_users": count,
            "visualization": {
                "type": "gauge",
                "data": [{
                    "name": "在线用户",
                    "value": count,
                    "min": 0,
                    "max": 100,  # 可以根据需要调整
                    "format": "{value}人"
                }]
            }
        })
        
        origin = request.headers.get('Origin')
        if origin in origins:
            response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response

    except Exception as e:
        return handle_error(e)
    


@app.route('/download_output')
async def download_output():
    job_id = request.args.get("job_id")
    stage_name = request.args.get("node_id")
    output_idx = int(request.args.get("output_idx"))

    try:
        data = await load_output(job_id, stage_name, output_idx)
    except Exception as e:
        return handle_error(e)
    
    # 构建二进制响应
    if isinstance(data, pl.DataFrame):
        # DataFrame转parquet二进制流
        buffer = io.BytesIO()
        data.write_parquet(buffer)
        buffer.seek(0)
        binary_data = buffer.getvalue()
        mimetype = 'application/octet-stream'
        filename = f"{stage_name}_output_{output_idx}.parquet"
    elif isinstance(data, dict) and "model" in data.keys():
        model_type = data["type"]
        model = data["model"]
        if model_type == "XGB":
            # XGB模型转json格式
            model_type = bytes(model.save_raw())
            binary_data = model_type
            mimetype = 'application/octet-stream'
            filename = f"{stage_name}_model.bin"
        elif model_type == "SQL":
            binary_data = model
            mimetype = 'text/plain'
            filename = f"{stage_name}_model.sql"
    elif isinstance(data, dict) and data["type"] == "file":
        if data["file_type"] == "excel":
            binary_data = open(data["file_path"], "rb").read()
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            filename = f"{stage_name}_report.xlsx"
    else:
        # 其他数据类型转pickle二进制流
        binary_data = pickle.dumps(data)
        mimetype = 'application/octet-stream'
        filename = f"{stage_name}_output_{output_idx}.pickle"

    # 返回二进制流响应
    response = Response(
        binary_data,
        mimetype=mimetype,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": mimetype,
            "Access-Control-Expose-Headers": "Content-Disposition",
        }
    )
    
    # 添加CORS头
    origin = request.headers.get('Origin')
    if origin in origins:
        response.headers.add("Access-Control-Allow-Origin", origin)
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response


if __name__ == "__main__":
    user_counter = UserCountActor.remote()
    # 定时清除缓存
    cleaner = Cleaner.remote()
    cleaner.run.remote()
    app.debug = False
    app.run(host='0.0.0.0', port=4242)
