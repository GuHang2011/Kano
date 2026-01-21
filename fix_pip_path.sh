#!/bin/bash

echo "🔧 修复pip路径问题..."

# 激活kano环境
conda activate kano

# 修复PATH
export PATH="$CONDA_PREFIX/bin:$PATH"
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "\.local/bin" | tr '\n' ':' | sed 's/:$//' | sed 's/^://')

echo "📝 修复后的环境："
echo "Python: $(which python)"
echo "Pip: $(which pip)"
echo "Conda prefix: $CONDA_PREFIX"

# 确保conda环境有pip
conda install pip -y

# 重新安装关键包到正确位置
echo "📦 重新安装numpy到conda环境..."
pip install --force-reinstall numpy==1.20.3

# 验证
echo "🔍 验证安装："
python -c "
import numpy
print(f'NumPy版本: {numpy.__version__}')
print(f'NumPy路径: {numpy.__file__}')
if '/home/zhangyue/anaconda3/envs/kano/' in numpy.__file__:
    print('✅ NumPy正确安装在conda环境中')
else:
    print('❌ NumPy仍在错误位置')
"

echo "✅ 修复完成！现在可以运行训练了。"
