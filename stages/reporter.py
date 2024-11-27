import os
import pandas as pd
from dags.stage import CustomStage
from stages.hdfs import hdfs_download, CACHE_PATH


class AnalysisReporter(CustomStage):

    def __init__(self, experiment_name: str, date: str, recover: bool = False):
        super().__init__(n_outputs=1)
        self.experiment_name = experiment_name
        self.date = date
        self.recover = recover
        self.base_path = "/projects/growth/test/analysis_tools/alg_label/memAlgCouponABAutoReportBaseV1/"
        self.csv_path = os.path.join(self.base_path, self.experiment_name, self.date)
        
    def forward(self):
        local_path = hdfs_download(self.logger, self.csv_path, self.recover)
        df = pd.read_csv(local_path)
        color_cols = [col for col in df.columns if "diff" in col]
        
        # 计算每列的最大绝对值
        max_abs_values = {col: df[col].abs().max() for col in color_cols}
        
        def color_negative_red(val, col):
            """
            根据数值设置颜色:
            - 正值: 绿色 (#00FF00)
            - 负值: 红色 (#FF0000)
            透明度根据该列最大绝对值计算
            """
            if pd.isna(val):
                return ''
            
            # 使用当前列的最大绝对值作为分母
            alpha = min(abs(val) / max_abs_values[col], 1.0)
            
            if val > 0:
                return f'background-color: rgba(0, 255, 0, {alpha})'
            else:
                return f'background-color: rgba(255, 0, 0, {alpha})'
        
        # 为每列分别应用样式
        styled_df = df.style
        for col in color_cols:
            styled_df = styled_df.applymap(
                lambda x, c=col: color_negative_red(x, c), 
                subset=[col]
            )
        
        output_path = os.path.join(CACHE_PATH, self._job_id, "out_files", f"{self.experiment_name}_{self.date}_report.xlsx")
        styled_df.to_excel(output_path, index=False, engine='openpyxl')
        
        return {"type": "file", "file_type": "excel", "file_path": output_path}
    