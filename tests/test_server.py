import requests
import json


def test_rerun_graph():
    json_file_path = "tests/test.json"
    server_url = "http://127.0.0.1:4242/rerun_graph"  # 假设Flask服务器运行在5000端口

    # 读取JSON文件
    with open(json_file_path, "r") as f:
        graph_json_str = f.read()

    # 准备请求数据
    data = {"data": graph_json_str}
    headers = {"Content-Type": "application/json"}

    # 发送POST请求
    response = requests.post(server_url, json=data, headers=headers)

    # 检查响应
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.text}")

    if response.status_code == 200:
        print("图表重新运行成功")
    else:
        print(f"错误: {response.status_code}")


if __name__ == "__main__":
    test_rerun_graph()
