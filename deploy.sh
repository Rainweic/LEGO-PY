#/bin/bash

ray start --head --dashboard-host "0.0.0.0"

export PYTHONPATH=$(pwd)
/data/soft/anaconda3/envs/py310/bin/python web/server.py