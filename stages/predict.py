import polars as pl
from dags.stage import CustomStage
from stages.xgb import XGB


class Predict(CustomStage):

    def __init__(self):
        super().__init__(n_outputs=1)

    def forward(self, model, lf: pl.LazyFrame):

        model_type = model['type']
        
        if model_type == "XGB":
            return XGB.predict(model=model, lf=lf)
