from dags.stage import CustomStage


class ExportModel(CustomStage):

    def __init__(self):
        super().__init__(n_outputs=0)
        
    def forward(self, model):
    
        model_type = model['type']
        model = model['model']

        if model_type == 'XGB':
            pass
        