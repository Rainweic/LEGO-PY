import pandas as pd

from dags.stage import stage

@stage
def case_when() -> pd.DataFrame:
    pass