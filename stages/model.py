import os
import uuid
from dags.stage import CustomStage


class ExportModel(CustomStage):

    def __init__(self):
        super().__init__(n_outputs=0)

    def forward(self, model):
    
        model_type = model['type']
        model = model['model']

        # 保存到临时文件目录
        model_name = f"{model_type}_{uuid.uuid4()}.bin"
        model_dir = os.path.join('cache', self.job_id, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_name = os.path.join(model_dir, model_name)

        self.logger.info(f"模型将保存到: {model_name}")

        if model_type == 'XGB':
            model.save_model(model_name)
        