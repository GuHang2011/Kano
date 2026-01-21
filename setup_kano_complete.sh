#!/bin/bash
# KANO完整环境配置脚本 - WSL版本
set -e
echo "🧬 KANO完整环境配置开始..."
echo "适用于蛋白质预测模型和生物信息学分析"
echo "=================================================="
# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
# 日志函数
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
# 检查系统依赖
check_system_deps() {
    log_info "检查系统依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装，请先安装Python3"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    log_success "Python版本: $python_version"
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        log_warning "pip3 未找到，尝试安装..."
        sudo apt update
        sudo apt install -y python3-pip
    fi
    
    # 检查conda
    if ! command -v conda &> /dev/null; then
        log_error "未检测到conda，请先安装miniconda或anaconda"
        exit 1
    fi
    
    log_success "系统依赖检查完成"
}
# 清理旧环境
cleanup_old_env() {
    log_info "清理旧环境..."
    
    # 删除kano环境如果存在
    if conda env list | grep -q "^kano "; then
        log_warning "发现旧的kano环境，正在删除..."
        conda env remove -n kano -y
    fi
    
    log_success "环境清理完成"
}
# 创建conda环境
create_conda_env() {
    log_info "创建conda环境 'kano'..."
    
    # 创建新环境
    conda create -n kano python=3.8 -y
    
    # 设置conda hook
    eval "$(conda shell.bash hook)"
    
    # 激活环境
    conda activate kano
    
    # 验证环境
    if [[ "$CONDA_DEFAULT_ENV" == "kano" ]]; then
        log_success "环境 'kano' 创建并激活成功"
    else
        log_error "环境激活失败"
        exit 1
    fi
}
# 安装PyTorch
install_pytorch() {
    log_info "安装PyTorch..."
    
    # 检测CUDA
    if command -v nvidia-smi &> /dev/null; then
        log_info "检测到NVIDIA GPU，安装CUDA版本"
        pip install torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
    else
        log_info "安装CPU版本PyTorch"
        pip install torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    fi
    
    # 安装torch扩展包
    log_info "安装torch扩展包..."
    pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
    pip install torch-geometric
    
    log_success "PyTorch安装完成"
}
# 安装科学计算包
install_scientific_packages() {
    log_info "安装科学计算包..."
    
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        numpy==1.20.3 \
        scipy \
        pandas \
        matplotlib \
        seaborn \
        scikit-learn==0.24.2 \
        networkx \
        plotly
    
    log_success "科学计算包安装完成"
}
# 安装化学信息学包
install_cheminformatics() {
    log_info "安装化学信息学包..."
    
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        rdkit-pypi \
        mordred \
        chempy
    
    log_success "化学信息学包安装完成"
}
# 安装生物信息学包
install_bioinformatics() {
    log_info "安装生物信息学包..."
    
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        biopython \
        bioservices \
        prody
    
    log_success "生物信息学包安装完成"
}
# 安装机器学习包
install_ml_packages() {
    log_info "安装机器学习包..."
    
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        xgboost \
        lightgbm \
        optuna
    
    log_success "机器学习包安装完成"
}
# 安装本体和知识图谱包
install_ontology_packages() {
    log_info "安装本体和知识图谱包..."
    
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "Owlready2==0.37" \
        "rdflib>=4.2.2" \
        "Click>=7.0" \
        "pyparsing==2.4.7" \
        "owl2vec-star==0.2.1" \
        "gensim==4.2.0"
    
    log_success "本体和知识图谱包安装完成"
}
# 安装开发工具
install_dev_tools() {
    log_info "安装开发工具..."
    
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        jupyter \
        jupyterlab \
        tensorboard \
        tqdm \
        rich \
        click
    
    log_success "开发工具安装完成"
}
# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    python -c "
import sys
print('Python版本:', sys.version)
print('=' * 50)
packages = {
    'torch': 'PyTorch',
    'torch_geometric': 'PyTorch Geometric', 
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'sklearn': 'scikit-learn',
    'rdkit': 'RDKit',
    'Bio': 'Biopython',
    'networkx': 'NetworkX',
    'gensim': 'Gensim',
    'owlready2': 'Owlready2',
    'owl2vec_star': 'OWL2Vec-Star',
    'xgboost': 'XGBoost',
    'optuna': 'Optuna',
    'tqdm': 'tqdm',
    'jupyter': 'Jupyter'
}
success_count = 0
total_count = len(packages)
for module, name in packages.items():
    try:
        if module == 'torch':
            import torch
            print(f'✅ {name}: {torch.__version__}')
        elif module == 'Bio':
            import Bio
            print(f'✅ {name}: {Bio.__version__}')
        else:
            __import__(module)
            print(f'✅ {name}: OK')
        success_count += 1
    except ImportError:
        print(f'❌ {name}: 导入失败')
    except Exception as e:
        print(f'⚠️ {name}: 部分功能可用')
        success_count += 0.5
print('=' * 50)
print(f'安装成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)')
# GPU检查
try:
    import torch
    print(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA设备数量: {torch.cuda.device_count()}')
except:
    pass
"
    
    log_success "安装验证完成"
}
# 创建项目文件
create_project_files() {
    log_info "创建项目文件..."
    
    # 创建目录
    mkdir -p {data,models,notebooks,scripts,results,logs,configs}
    
    # 创建激活脚本
    cat > activate.sh << 'ACTIVATE_END'
#!/bin/bash
echo "激活KANO环境..."
eval "$(conda shell.bash hook)"
conda activate kano
echo "✅ 环境已激活: $(conda info --envs | grep '*')"
echo "Python路径: $(which python)"
echo "Python版本: $(python --version)"
ACTIVATE_END
    
    chmod +x activate.sh
    
    # 保存环境信息
    conda list --export > requirements_conda.txt
    pip freeze > requirements_pip.txt
    
    log_success "项目文件创建完成"
}
# 主函数
main() {
    log_info "开始KANO完整环境配置..."
    
    check_system_deps
    cleanup_old_env
    create_conda_env
    install_pytorch
    install_scientific_packages  
    install_cheminformatics
    install_bioinformatics
    install_ml_packages
    install_ontology_packages
    install_dev_tools
    verify_installation
    create_project_files
    
    echo ""
    echo "🎉🎉🎉 KANO环境配置完成！ 🎉🎉🎉"
    echo ""
    echo "📋 接下来的步骤："
    echo "1. 激活环境:"
    echo "   conda activate kano"
    echo "   # 或者运行: bash activate.sh"
    echo ""
    echo "2. 启动Jupyter Lab:"
    echo "   jupyter lab"
    echo ""
    echo "🧬 现在可以开始蛋白质预测模型开发了！"
    echo ""
}
# 执行主函数
main "$@"
