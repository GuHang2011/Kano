#!/bin/bash
# KANO环境配置脚本 - 修复版本
set -e
echo "🧬 KANO环境配置开始..."
echo "=================================================="
# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}
log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}
log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}
log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}
# 检查conda环境
check_conda() {
    if ! command -v conda &> /dev/null; then
        log_error "未检测到conda，请先安装miniconda或anaconda"
        exit 1
    fi
    log_success "检测到conda"
}
# 创建conda环境
create_env() {
    log_info "创建conda环境..."
    
    # 删除旧环境（如果存在）
    if conda env list | grep -q "^kano "; then
        log_warning "删除旧环境..."
        conda env remove -n kano -y
    fi
    
    # 创建新环境
    conda create -n kano python=3.8 -y
    
    # 激活环境
    eval "$(conda shell.bash hook)"
    conda activate kano
    
    log_success "环境创建完成"
}
# 升级pip
upgrade_pip() {
    log_info "升级pip..."
    python -m pip install --upgrade pip
    log_success "pip升级完成"
}
# 安装PyTorch
install_pytorch() {
    log_info "安装PyTorch..."
    
    if command -v nvidia-smi &> /dev/null; then
        log_info "检测到GPU，安装CUDA版本"
        pip install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    else
        log_info "安装CPU版本"
        pip install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # 安装torch扩展
    log_info "安装torch扩展..."
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
    
    log_success "PyTorch安装完成"
}
# 安装核心科学计算包
install_core_packages() {
    log_info "安装核心包..."
    
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        numpy==1.20.3 \
        scipy \
        pandas \
        matplotlib \
        seaborn \
        scikit-learn==0.24.2 \
        networkx \
        tqdm \
        plotly
    
    log_success "核心包安装完成"
}
# 安装化学和生物信息学包
install_bio_chem_packages() {
    log_info "安装生物化学包..."
    
    # 化学信息学
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rdkit-pypi
    
    # 生物信息学
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple biopython
    
    # 机器学习
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        xgboost \
        lightgbm \
        optuna
    
    log_success "生物化学包安装完成"
}
# 安装本体和知识图谱包（修复版）
install_ontology_packages() {
    log_info "安装本体和知识图谱包..."
    
    # 基础本体包
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "Owlready2==0.37" \
        "rdflib>=4.2.2" \
        "gensim==4.2.0"
    
    # 尝试安装OWL2Vec-Star（从GitHub）
    log_info "尝试安装OWL2Vec-Star..."
    pip install git+https://github.com/KRR-Oxford/OWL2Vec-Star.git || {
        log_warning "OWL2Vec-Star安装失败，将跳过此包"
        log_info "可以后续手动安装: pip install git+https://github.com/KRR-Oxford/OWL2Vec-Star.git"
    }
    
    log_success "本体包安装完成"
}
# 安装开发工具
install_dev_tools() {
    log_info "安装开发工具..."
    
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        jupyter \
        jupyterlab \
        tensorboard \
        rich \
        click
    
    log_success "开发工具安装完成"
}
# 安装KANO特定依赖
install_kano_deps() {
    log_info "安装KANO项目依赖..."
    
    # 检查是否存在requirements.txt或其他依赖文件
    if [[ -f "requirements.txt" ]]; then
        log_info "发现requirements.txt，安装项目依赖..."
        pip install -r requirements.txt
    fi
    
    # 安装chemprop（如果存在chemprop目录）
    if [[ -d "chemprop" ]]; then
        log_info "安装chemprop..."
        cd chemprop
        pip install -e .
        cd ..
    fi
    
    log_success "KANO依赖安装完成"
}
# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    python -c "
import sys
print('🐍 Python版本:', sys.version.split()[0])
print('📍 Python路径:', sys.executable)
print('🌟 环境名称: $CONDA_DEFAULT_ENV')
print('=' * 60)
# 核心包测试
packages_to_test = [
    ('torch', 'PyTorch'),
    ('numpy', 'NumPy'), 
    ('pandas', 'Pandas'),
    ('matplotlib', 'Matplotlib'),
    ('sklearn', 'scikit-learn'),
    ('networkx', 'NetworkX'),
    ('scipy', 'SciPy'),
    ('seaborn', 'Seaborn'),
    ('rdkit', 'RDKit'),
    ('Bio', 'Biopython'),
    ('gensim', 'Gensim'),
    ('owlready2', 'Owlready2'),
    ('xgboost', 'XGBoost'),
    ('optuna', 'Optuna'),
    ('jupyter', 'Jupyter'),
    ('tqdm', 'tqdm'),
    ('rich', 'Rich')
]
success = 0
total = len(packages_to_test)
for module, name in packages_to_test:
    try:
        if module == 'torch':
            import torch
            print(f'✅ {name}: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
        elif module == 'Bio':
            import Bio
            print(f'✅ {name}: {Bio.__version__}')
        elif module == 'sklearn':
            import sklearn
            print(f'✅ {name}: {sklearn.__version__}')
        else:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'OK')
            print(f'✅ {name}: {version}')
        success += 1
    except ImportError:
        print(f'❌ {name}: 未安装')
    except Exception as e:
        print(f'⚠️  {name}: 部分可用')
        success += 0.5
print('=' * 60)
print(f'📊 安装成功率: {success}/{total} ({success/total*100:.1f}%)')
# 特殊测试
try:
    import torch
    print(f'🔥 PyTorch CUDA: {\"可用\" if torch.cuda.is_available() else \"不可用\"}')
except:
    pass
"
    
    log_success "验证完成"
}
# 创建便捷脚本
create_scripts() {
    log_info "创建便捷脚本..."
    
    # 创建激活脚本
    cat > activate_kano.sh << 'EOF'
#!/bin/bash
echo "🧬 激活KANO环境..."
eval "$(conda shell.bash hook)"
conda activate kano
echo "✅ 环境已激活"
echo "📍 Python: $(which python)"
echo "🐍 版本: $(python --version)"
EOF
    
    # 创建启动Jupyter的脚本
    cat > start_jupyter.sh << 'EOF'
#!/bin/bash
echo "🚀 启动Jupyter Lab..."
eval "$(conda shell.bash hook)"
conda activate kano
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
EOF
    
    # 创建环境信息脚本
    cat > env_info.sh << 'EOF'
#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate kano
echo "📋 KANO环境信息"
echo "=================="
echo "🐍 Python: $(python --version)"
echo "📍 位置: $(which python)"
echo "🌟 Conda环境: $CONDA_DEFAULT_ENV"
echo ""
echo "📦 主要包版本:"
python -c "
packages = ['torch', 'numpy', 'pandas', 'sklearn', 'rdkit']
for pkg in packages:
    try:
        if pkg == 'sklearn':
            import sklearn as mod
        else:
            mod = __import__(pkg)
        version = getattr(mod, '__version__', 'Unknown')
        print(f'  {pkg}: {version}')
    except:
        print(f'  {pkg}: 未安装')
"
EOF
    
    chmod +x activate_kano.sh start_jupyter.sh env_info.sh
    
    log_success "便捷脚本创建完成"
}
# 主函数
main() {
    log_info "🚀 开始KANO环境配置..."
    
    check_conda
    create_env
    upgrade_pip
    install_pytorch
    install_core_packages
    install_bio_chem_packages
    install_ontology_packages
    install_dev_tools
    install_kano_deps
    verify_installation
    create_scripts
    
    echo ""
    echo "🎉🎉🎉 KANO环境配置成功！ 🎉🎉🎉"
    echo ""
    echo "📋 快速开始："
    echo "  激活环境:    bash activate_kano.sh"
    echo "  启动Jupyter: bash start_jupyter.sh" 
    echo "  查看信息:    bash env_info.sh"
    echo ""
    echo "🔬 环境已就绪，开始你的蛋白质预测研究吧！"
    echo ""
}
# 执行
main "$@"
