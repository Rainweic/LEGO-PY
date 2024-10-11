import json
import yaml


def json2yaml(str_json):

    infos = json.loads(str_json)['cells']

    nodes = {}

    # 生成node
    for item in infos:
        if item.get("shape", None) == "dag-node":

            # print(item["id"])
            # print(item)
            
            nodes[item["id"]] = {
                "name": item["id"],
                "label": item["data"]["label"],
                "stage": item["data"]["key"],
                "args": item["data"].get("args", {}),
                "collect_result": item.get("collect_result", False)
            }

            # print(nodes[item["id"]])

    # print(nodes)

    # 遍历边
    for item in infos:
        if item.get("shape", None) == "dag-edge":

            source_node_name = item["source"]["cell"]
            target_node_name = item["target"]["cell"]

            # 设置after
            if ("after" in nodes[target_node_name]) and (target_node_name not in nodes[target_node_name]["after"]):
                nodes[target_node_name]["after"].append(source_node_name)
            else:
                nodes[target_node_name]["after"] = [source_node_name]

            # 设置inputs
            input_idx = int(item['source']['port'].split('-')[1])
            input_name = f"{source_node_name}.{input_idx}"
            if "inputs" in nodes[target_node_name]:
                nodes[target_node_name]["inputs"].append(input_name)
            else:
                nodes[target_node_name]["inputs"] = [input_name]

            # print(nodes[source_node_name])
            # print(nodes[target_node_name])

    # 这里的stages需要有一个先后顺序，要不然不能正常生成pipeline
    def topological_sort(nodes):
        visited = set()
        result = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for dep in nodes[node].get('after', []):
                dfs(dep)
            result.append(node)
        
        for node in nodes:
            dfs(node)
        
        return result
    sorted_nodes = topological_sort(nodes)
    stages = [{nodes[node]['stage']: nodes[node]} for node in sorted_nodes]
    
    yaml_content = {
        "global_args": {},
        "pipeline": [{
            "name": "",
            "args": {
                "visualize": True,
                "save_dags": True,
                "force_rerun": False
            },
            "stages": stages
        }]
    }
    
    return yaml.dump(yaml_content, allow_unicode=True)
