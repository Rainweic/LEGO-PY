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
        local_path = os.path.join(local_path, self.date)
        local_path_csv = [f for f in os.listdir(local_path) if f.endswith('.csv')][0]
        df = pd.read_csv(os.path.join(local_path, local_path_csv))

        df["diff_饮品用户数百分比"] = df["diff_饮品用户数"] / df["对照组饮品用户数"]
        df["diff_饮品数百分比"] = df["diff_饮品数"] / df["对照组饮品数"]
        color_cols = ["diff_饮品用户数", "diff_饮品数", "diff_饮品用户数百分比", "diff_饮品数百分比"]
      
        styled_df = df.style.format('{:.3f}', subset=color_cols, na_rep="")\
            .bar(subset=color_cols, align=0, vmin=-2.5, vmax=2.5, cmap="bwr",
                 props="width: 120px; position: relative; padding: 0px 4px;")\
            .set_properties(**{'color': 'black', 'position': 'relative', 'z-index': 1})\
            .text_gradient(subset=color_cols, cmap="bwr", vmin=-2.5, vmax=2.5)
        
        save_dir = os.path.join(CACHE_PATH, self._job_id, "out_files")
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"{self.experiment_name}_{self.date}_report.xlsx")
        self.logger.info(f"Saving report to {output_path}")
        styled_df.to_excel(output_path, index=False, engine='openpyxl')
        
        return {"type": "file", "file_type": "excel", "file_path": output_path}
    