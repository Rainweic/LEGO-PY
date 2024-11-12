import polars as pl
from dags.stage import CustomStage
from stages.xgb import XGB
from stages.score_card import ScoreCard


class Predict(CustomStage):

    def __init__(self, threshold=0.5):
        super().__init__(n_outputs=1)
        self.threshold = threshold

    def forward(self, model, lf: pl.LazyFrame):

        if self.threshold > 1 or self.threshold < 0:
            self.logger.error("阈值的取值范围应该是(0, 1)")
            raise TypeError("阈值的取值范围应该是(0, 1)")

        model_type = model['type']
        
        if model_type == "XGB":
            result = XGB.predict(model=model, lf=lf)
        elif model_type == "ScoreCard":
            result = ScoreCard.predict(model=model, data=lf)

        if "y_score" in result.columns:
            result = result.with_columns(
                pl.col("*"),
                pl.when(pl.col("y_score") > self.threshold).then(1).otherwise(0).alias("y_pred")
            )
        
        return result