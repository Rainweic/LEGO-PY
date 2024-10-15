import json
import tempfile
import asyncio
import argparse

from utils.convert import json2yaml
from dags.parser import load_pipelines_from_yaml

# 读取画布导出的json文件，自动生成计算图并进行计算

async def main():

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="读取YAML文件，自动生成计算图并进行计算")
    parser.add_argument('-p', '--path', type=str, help='YAML文件的路径')

    args = parser.parse_args()

    with open(args.path, "r") as f:
        graph_json_str = json.load(f)
    
    graph_yaml, job_id = json2yaml(graph_json_str, force_rerun=False)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_file.write(graph_yaml)
        temp_file_path = temp_file.name
    
    print(f"temp yaml: {temp_file_path}")
    print(f"job id: {job_id}")

    await load_pipelines_from_yaml(temp_file_path)

if __name__ == "__main__":
    asyncio.run(main=main())