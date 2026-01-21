#!/bin/bash

echo "🧬 KANO环境安装脚本 - 清华源版本"
echo "使用清华镜像源和python -m pip安装"
echo "====================================="

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# 检查conda是否安装
check_conda() {
    if ! command -v conda &> /dev/null; then
        log_error "Conda未找到，请先安装Anaconda或Miniconda"
        exit 1
    fi
    log_success "Conda已安装: $(conda --version)"
}

# 激活环境或创建新环境
setup_environment() {
    log_info "设置kano环境..."
    
    eval "$(conda shell.bash hook)"
    
    # 检查环境是否存在
    if conda env list | grep -q "kano"; then
        log_info "激活现有的kano环境..."
        conda activate kano
    else
        log_info "创建新的kano环境..."
        conda create -n kano python=3.8 -y
        conda activate kano
    fi
    
    if [[ "$CONDA_DEFAULT_ENV" != "kano" ]]; then
        log_error "环境激活失败"
        exit 1
    fi
    
    log_success "当前环境: $CONDA_DEFAULT_ENV"
    log_success "Python版本: $(python --version)"
}

# 更新pip并配置清华源
setup_pip() {
    log_info "配置pip和清华源..."
    
    # 创建pip配置目录
    mkdir -p ~/.pip
    
    # 配置pip使用清华源
    cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
EOF
    
    # 更新pip
    log_info "更新pip..."
    python -m pip install --upgrade pip
    
    log_success "pip配置完成"
    python -m pip --version
}

# 安装conda包
install_conda_packages() {
    log_info "安装conda包..."
    
    conda install -c conda-forge -y \
        numpy \
        pandas \
        matplotlib \
        scipy \
        scikit-learn \
        jupyter \
        notebook \
        jupyterlab \
        tqdm \
        networkx \
        seaborn
    
    log_success "Conda包安装完成"
}

# 安装PyTorch
install_pytorch() {
    log_info "安装PyTorch..."
    
    # CPU版本的PyTorch（更稳定）
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    log_success "PyTorch安装完成"
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
}

# 安装核心pip包
install_core_packages() {
    log_info "安装核心pip包..."
    
    # 核心包列表
    core_packages=(
        "rdkit-pypi"
        "biopython" 
        "gensim"
        "owlready2"
        "xgboost"
        "lightgbm"
        "optuna"
        "rich"
        "plotly"
        "tensorboard"
        "flask"
        "werkzeug"
        "jinja2"
        "click"
    )
    
    for package in "${core_packages[@]}"; do
        log_info "安装 $package..."
        python -m pip install "$package" || {
            log_warning "$package 安装失败，跳过..."
            continue
        }
        log_success "$package 安装成功"
    done
}

# 安装PyTorch Geometric
install_torch_geometric() {
    log_info "安装PyTorch Geometric..."
    
    # 获取PyTorch版本
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    log_info "检测到PyTorch版本: $TORCH_VERSION"
    
    # 安装torch-geometric相关包
    python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html || {
        log_warning "PyTorch Geometric安装失败，使用基础版本..."
        python -m pip install torch-geometric
    }
    
    log_success "PyTorch Geometric安装完成"
}

# 尝试安装可选包
install_optional_packages() {
    log_info "安装可选包..."
    
    optional_packages=(
        "hyperopt"
        "chemprop"
        "transformers"
        "datasets"
    )
    
    for package in "${optional_packages[@]}"; do
        log_info "尝试安装 $package..."
        python -m pip install "$package" || {
            log_warning "$package 安装失败，这是可选包，可以跳过"
            continue
        }
        log_success "$package 安装成功"
    done
}

# 处理OWL2Vec-Star
handle_owl2vec() {
    log_info "尝试安装OWL2Vec-Star..."
    
    # 尝试从GitHub安装
    python -m pip install git+https://github.com/KRR-Oxford/OWL2Vec-Star.git || {
        log_warning "OWL2Vec-Star从GitHub安装失败，这是可选包"
        
        # 创建占位符，避免import错误
        mkdir -p kano_placeholder/owl2vec_star
        cat > kano_placeholder/__init__.py << 'EOF'
# OWL2Vec-Star placeholder
def get_owl2vec_embeddings(*args, **kwargs):
    raise NotImplementedError("OWL2Vec-Star not installed. Install with: pip install git+https://github.com/KRR-Oxford/OWL2Vec-Star.git")
EOF
        
        log_info "已创建OWL2Vec-Star占位符"
    }
}

# 环境验证
verify_installation() {
    log_info "验证安装..."
    
    python -c "
import sys
print(f'🐍 Python: {sys.version.split()[0]}')
print(f'📍 环境: $CONDA_DEFAULT_ENV')
print(f'💾 路径: {sys.executable}')
print()

# 核心包测试
test_packages = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'sklearn': 'scikit-learn',
    'rdkit': 'RDKit',
    'Bio': 'Biopython',
    'networkx': 'NetworkX',
    'gensim': 'Gensim',
    'owlready2': 'Owlready2',
    'xgboost': 'XGBoost',
    'optuna': 'Optuna',
    'jupyter': 'Jupyter',
    'rich': 'Rich'
}

print('📦 包验证结果:')
success_count = 0
total_count = len(test_packages)

for module, name in test_packages.items():
    try:
        if module == 'sklearn':
            import sklearn
            print(f'  ✅ {name}: {sklearn.__version__}')
        elif module == 'Bio':
            import Bio
            print(f'  ✅ {name}: {Bio.__version__}')
        elif module == 'rdkit':
            from rdkit import Chem
            print(f'  ✅ {name}: OK')
        else:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'OK')
            print(f'  ✅ {name}: {version}')
        success_count += 1
    except ImportError as e:
        print(f'  ❌ {name}: 未安装')
    except Exception as e:
        print(f'  ⚠️  {name}: 部分可用')
        success_count += 0.5

print()
print(f'📊 安装成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)')

# GPU检查
try:
    import torch
    print(f'🔥 CUDA可用: {\"是\" if torch.cuda.is_available() else \"否 (CPU版本)\"}')
except:
    pass
"
}

# 创建便捷脚本
create_helper_scripts() {
    log_info "创建便捷脚本..."
    
    # 激活环境脚本
    cat > activate_kano.sh << 'EOF'
#!/bin/bash
echo "🧬 激活KANO环境..."
eval "$(conda shell.bash hook)"
conda activate kano
echo "✅ 环境已激活: $CONDA_DEFAULT_ENV"
echo "🐍 Python: $(python --version)"
echo "📍 路径: $(which python)"
EOF
    chmod +x activate_kano.sh
    
    # Jupyter启动脚本
    cat > start_jupyter.sh << 'EOF'
#!/bin/bash
echo "🚀 启动Jupyter Lab..."
eval "$(conda shell.bash hook)"
conda activate kano
echo "🌐 访问地址: http://localhost:8888"
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
EOF
    chmod +x start_jupyter.sh
    
    # 环境测试脚本
    cat > test_kano.sh << 'EOF'
#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate kano

echo "🧪 KANO环境全面测试"
echo "==================="

# 基础测试
python -c "
import sys, os
print(f'🐍 Python: {sys.version}')
print(f'📍 环境: \$CONDA_DEFAULT_ENV') 
print(f'💾 执行路径: {sys.executable}')
print(f'📂 工作目录: {os.getcwd()}')
print()

# 导入测试
test_imports = [
    ('torch', 'PyTorch深度学习'),
    ('numpy', '数值计算'),
    ('pandas', '数据处理'),
    ('matplotlib', '绘图'),
    ('sklearn', '机器学习'),
    ('rdkit.Chem', 'RDKit化学'),
    ('Bio', '生物信息'),
    ('networkx', '图论'),
    ('gensim', '自然语言处理'),
    ('xgboost', 'XGBoost'),
    ('jupyter', 'Jupyter'),
]

print('🔍 功能模块测试:')
for module, desc in test_imports:
    try:
        __import__(module)
        print(f'  ✅ {desc}: 正常')
    except ImportError:
        print(f'  ❌ {desc}: 缺失') 
    except Exception as e:
        print(f'  ⚠️  {desc}: 异常({str(e)[:30]})')

# 简单功能测试
print()
print('⚡ 功能测试:')

try:
    import torch
    x = torch.randn(3, 3)
    print(f'  ✅ PyTorch张量运算: {x.shape}')
except:
    print('  ❌ PyTorch张量运算失败')

try:
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(np.random.randn(5, 3))
    print(f'  ✅ Pandas数据框: {df.shape}')
except:
    print('  ❌ Pandas数据框失败')

try:
    from rdkit import Chem
    mol = Chem.MolFromSmiles('CCO')
    print(f'  ✅ RDKit分子解析: {mol.GetNumAtoms()}原子')
except:
    print('  ❌ RDKit分子解析失败')

print()
print('🎉 环境测试完成！')
"
EOF
    chmod +x test_kano.sh
    
    log_success "便捷脚本创建完成"
}

# 主安装流程
main() {
    echo "开始KANO环境安装..."
    
    check_conda
    setup_environment  
    setup_pip
    install_conda_packages
    install_pytorch
    install_core_packages
    install_torch_geometric
    install_optional_packages
    handle_owl2vec
    verify_installation
    create_helper_scripts
    
    echo ""
    echo "🎉 KANO环境安装完成！"
    echo ""
    echo "📋 接下来你可以："
    echo "  激活环境:    bash activate_kano.sh"
    echo "  测试环境:    bash test_kano.sh" 
    echo "  启动Jupyter: bash start_jupyter.sh"
    echo ""
    echo "🔬 环境就绪，开始你的生物信息学研究吧！"
}

# 执行主函数
main "$@"
