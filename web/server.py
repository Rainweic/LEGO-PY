# -*- coding: utf-8 -*
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dags.cache import SQLiteCache
from dags.pipeline import StageStatus
from utils.convert import json2yaml
from dags.parser import load_pipelines_from_yaml

import os
import math
import uuid
import pickle
import logging
import asyncio
import traceback
import polars as pl
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


app = Flask(__name__)
origins = ["http://127.0.0.1:8000", "http://localhost:8000", "http://lego-ui:8000", "http://10.222.107.184:8000"]

CORS(app, resources={r"/rerun_graph": {"origins": origins}}, supports_credentials=True)


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
    graph_yaml, job_id = json2yaml(graph_json_str, force_rerun=force_rerun)
    config_dir = os.path.join('cache', job_id, 'config')
    os.makedirs(config_dir, exist_ok=True)
    temp_file_path = os.path.join(config_dir, 'graph_config.yaml')
    with open(temp_file_path, 'w') as temp_file:
        temp_file.write(graph_yaml)
    logging.info(f"YAML 内容已写入临时文件: {temp_file_path}")
    
    # 创建一个新的事件循环来运行 load_pipelines_from_yaml
    def run_in_new_loop(temp_file_path):
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            new_loop.run_until_complete(load_pipelines_from_yaml(temp_file_path))
        finally:
            new_loop.close()

    # 使用 ThreadPoolExecutor 在新线程中运行 load_pipelines_from_yaml
    executor = ThreadPoolExecutor()
    executor.submit(run_in_new_loop, temp_file_path)

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

            stage_names = request.json.get('stage_names')
            job_id = request.json.get("job_id")

            # 创建一个SQLiteCache实例
            cache = SQLiteCache()
            
            ret = {}
            for name in stage_names:
                # 从SQLite数据库中读取stage状态
                status = asyncio.run(cache.read(f"{job_id}_{name}"))
            
                if status is None:
                    status = 'running'
            
                # 将状态转换为StageStatus枚举
                stage_status = StageStatus(status)

                ret[name] = stage_status.value

            logging.info(f"Job id: {job_id}, status: {ret}")
                
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
        if os.path.exists(stage_log_path):
            with open(stage_log_path, "r") as f:
                log_content = f.read()
        elif os.path.exists(pipeline_log_path):
            with open(pipeline_log_path, "r") as f:
                log_content = f.read()
        else:
            msg = f"Stage、Pipeline日志文件均不存在, 路径：{base_path}"
            logging.info(msg)
            log_content = msg

        # 构建响应
        response = jsonify({"log": log_content})
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
                    if dtype in [pl.Float32, pl.Float64, pl.Int16, pl.Int32, pl.Int64, pl.Int8]:
                        return pl.col(col).fill_nan(None)
                    elif dtype == pl.Utf8:
                        return pl.col(col).fill_null("")  # 字符串类型的空值填充为空字符串
                    else:
                        return pl.col(col).fill_null(None)  # 其他类型的空值填充为None
                
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

        # response = jsonify({
        #                     "schema": [
        #                         {
        #                             "key": "mem_id",
        #                             "name": "mem_id",
        #                             "type": "Int64"
        #                         },
        #                         {
        #                             "key": "old_price_level",
        #                             "name": "old_price_level",
        #                             "type": "Int64"
        #                         }
        #                     ]
        #                 })
        
        origin = request.headers.get('Origin')
        if origin in origins:
            response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response
    except Exception as e:
        return handle_error(e)


# def bar_base():
#     from random import randrange
#     from pyecharts import options as opts
#     from pyecharts.charts import Bar
#     c = (
#         Bar()
#         .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
#         .add_yaxis("商家A", [randrange(0, 100) for _ in range(6)])
#         .add_yaxis("商家B", [randrange(0, 100) for _ in range(6)])
#         .set_global_opts(title_opts=opts.TitleOpts(title="Bar-基本示例", subtitle="我是副标题"))
#     )
#     summary = c.dump_options_with_quotes()
#     return {"hasData": True, "datas": [{"图表1": summary, "图表2": summary}]}


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
                items.append({
                    "name": item,
                    "path": item_path,
                    "type": "directory" if os.path.isdir(item_path) else "file",
                    "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None,
                    "modified": os.path.getmtime(item_path)
                })
        
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


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=4242)
