#!/bin/bash
echo "🧬 激活KANO环境..."
eval "$(conda shell.bash hook)"
conda activate kano
echo "✅ 环境已激活: $CONDA_DEFAULT_ENV"
echo "🐍 Python: $(python --version)"
echo "📍 路径: $(which python)"
