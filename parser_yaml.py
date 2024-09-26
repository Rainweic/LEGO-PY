import os
import argparse

from dags.parser import load_pipelines_from_yaml

# 读取yaml文件，自动生成计算图并进行计算

def main():

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="读取YAML文件，自动生成计算图并进行计算")
    parser.add_argument('-p', '--path', type=str, help='YAML文件的路径')

    args = parser.parse_args()

    load_pipelines_from_yaml(args.path)

if __name__ == "__main__":
    main()