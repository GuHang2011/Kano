#!/bin/bash
echo "🚀 启动Jupyter Lab..."
eval "$(conda shell.bash hook)"
conda activate kano
echo "🌐 访问地址: http://localhost:8888"
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
