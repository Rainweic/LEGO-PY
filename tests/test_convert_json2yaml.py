import json
from utils.convert import json2yaml


# test
if __name__ == "__main__":

    json_file_path = "tests/test.json"
    yaml_file_path = "tests/test.yaml"

    with open(json_file_path, "r",  encoding="utf-8") as f:
        json_str = f.read()

    yaml_str, _ = json2yaml(json_str)

    with open(yaml_file_path, "w", encoding="utf-8") as f:
        f.write(yaml_str)

    # 测试能否被正常解析
    import asyncio
    from dags.parser import load_pipelines_from_yaml
    asyncio.run(main=load_pipelines_from_yaml(yaml_file_path))
