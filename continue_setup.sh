#!/bin/bash

echo "🔄 继续完成KANO环境配置..."
echo "不会删除现有环境，只补充缺失的包"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'  
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

# 确保在kano环境中
log_info "激活kano环境..."
eval "$(conda shell.bash hook)"
conda activate kano

if [[ "$CONDA_DEFAULT_ENV" != "kano" ]]; then
    echo "❌ 请先激活kano环境: conda activate kano"
    exit 1
fi

log_success "当前环境: $CONDA_DEFAULT_ENV"

# 检查和安装缺失的包
install_missing_packages() {
    log_info "检查并安装缺失的包..."
    
    # 定义要检查的包
    declare -A packages=(
        ["torch"]="PyTorch"
        ["numpy"]="NumPy" 
        ["pandas"]="Pandas"
        ["matplotlib"]="Matplotlib"
        ["sklearn"]="scikit-learn"
        ["rdkit"]="RDKit"
        ["Bio"]="Biopython"
        ["networkx"]="NetworkX"
        ["gensim"]="Gensim"
        ["owlready2"]="Owlready2"
        ["xgboost"]="XGBoost"
        ["optuna"]="Optuna"
        ["jupyter"]="Jupyter"
        ["tqdm"]="tqdm"
        ["rich"]="Rich"
        ["seaborn"]="Seaborn"
        ["scipy"]="SciPy"
        ["lightgbm"]="LightGBM"
    )
    
    missing_packages=()
    
    # 检查每个包
    for module in "${!packages[@]}"; do
        if python -c "import $module" 2>/dev/null; then
            log_success "${packages[$module]} ✓"
        else
            log_warning "${packages[$module]} 缺失，将安装"
            case $module in
                "sklearn")
                    missing_packages+=("scikit-learn==0.24.2")
                    ;;
                "Bio")
                    missing_packages+=("biopython")
                    ;;
                "rdkit")
                    missing_packages+=("rdkit-pypi")
                    ;;
                *)
                    missing_packages+=("$module")
                    ;;
            esac
        fi
    done
    
    # 安装缺失的包
    if [ ${#missing_packages[@]} -gt 0 ]; then
        log_info "安装缺失的包: ${missing_packages[*]}"
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "${missing_packages[@]}"
    else
        log_success "所有核心包都已安装"
    fi
}

# 尝试安装torch-geometric相关（如果还没有）
install_torch_geometric() {
    log_info "检查PyTorch Geometric..."
    
    if python -c "import torch_geometric" 2>/dev/null; then
        log_success "PyTorch Geometric 已安装"
    else
        log_info "安装PyTorch Geometric..."
        pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
    fi
}

# 手动处理chemprop（如果需要）
handle_chemprop() {
    log_info "处理chemprop..."
    
    if [[ -d "chemprop" ]]; then
        cd chemprop
        if [[ -f "setup.py" ]] || [[ -f "pyproject.toml" ]]; then
            log_info "安装chemprop..."
            pip install -e .
        else
            log_warning "chemprop目录存在但无安装文件，跳过"
            # 尝试直接安装公开版本
            pip install chemprop 2>/dev/null || log_warning "chemprop公开版本安装失败"
        fi
        cd ..
    else
        log_info "尝试安装公开版chemprop..."
        pip install chemprop 2>/dev/null || log_warning "chemprop安装失败，可能不是必需的"
    fi
}

# 创建便捷脚本（如果不存在）
create_helper_scripts() {
    log_info "创建便捷脚本..."
    
    # 激活脚本
    if [[ ! -f "activate_kano.sh" ]]; then
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
        log_success "创建 activate_kano.sh"
    fi
    
    # Jupyter启动脚本
    if [[ ! -f "start_jupyter.sh" ]]; then
        cat > start_jupyter.sh << 'EOF'
#!/bin/bash
echo "🚀 启动Jupyter Lab..."
eval "$(conda shell.bash hook)"
conda activate kano
echo "🌐 访问地址: http://localhost:8888"
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
EOF
        chmod +x start_jupyter.sh
        log_success "创建 start_jupyter.sh"
    fi
    
    # 环境测试脚本
    if [[ ! -f "test_env.sh" ]]; then
        cat > test_env.sh << 'EOF'
#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate kano

echo "🧪 KANO环境测试"
echo "================"
python -c "
import sys
print(f'🐍 Python: {sys.version.split()[0]}')
print(f'📍 路径: {sys.executable}')
print(f'🌟 环境: \$CONDA_DEFAULT_ENV')
print()

# 快速测试
packages = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'pandas': 'Pandas', 
    'sklearn': 'scikit-learn',
    'rdkit': 'RDKit',
    'Bio': 'Biopython',
    'gensim': 'Gensim',
    'jupyter': 'Jupyter'
}

working = []
failed = []

for pkg, name in packages.items():
    try:
        __import__(pkg)
        working.append(name)
    except:
        failed.append(name)

print('✅ 工作正常:')
for pkg in working:
    print(f'  - {pkg}')

if failed:
    print()
    print('❌ 需要检查:')
    for pkg in failed:
        print(f'  - {pkg}')

print(f'\n📊 成功率: {len(working)}/{len(packages)} ({len(working)/len(packages)*100:.0f}%)')
"
EOF
        chmod +x test_env.sh
        log_success "创建 test_env.sh"
    fi
}

# 最终验证
final_verification() {
    log_info "最终验证..."
    
    echo "🧪 环境测试报告"
    echo "=================="
    
    python -c "
import sys
print(f'🐍 Python: {sys.version.split()[0]}')
print(f'📍 环境: \$CONDA_DEFAULT_ENV')
print(f'💾 位置: {sys.executable}')
print()

# 核心包测试
core_packages = [
    'torch', 'numpy', 'pandas', 'matplotlib', 
    'sklearn', 'scipy', 'networkx', 'tqdm'
]

bio_packages = [
    'rdkit', 'Bio', 'gensim', 'owlready2'
]

ml_packages = [
    'xgboost', 'optuna', 'lightgbm'
]

dev_packages = [
    'jupyter', 'rich'
]

def test_packages(pkg_list, category):
    print(f'📦 {category}:')
    success = 0
    for pkg in pkg_list:
        try:
            if pkg == 'sklearn':
                import sklearn
                print(f'  ✅ scikit-learn: {sklearn.__version__}')
            elif pkg == 'Bio':
                import Bio
                print(f'  ✅ Biopython: {Bio.__version__}')
            elif pkg == 'torch':
                import torch
                print(f'  ✅ PyTorch: {torch.__version__}')
            else:
                mod = __import__(pkg)
                version = getattr(mod, '__version__', 'OK')
                print(f'  ✅ {pkg}: {version}')
            success += 1
        except ImportError:
            print(f'  ❌ {pkg}: 未安装')
        except Exception as e:
            print(f'  ⚠️  {pkg}: 部分可用')
            success += 0.5
    return success, len(pkg_list)

s1, t1 = test_packages(core_packages, '核心包')
print()
s2, t2 = test_packages(bio_packages, '生物信息学')
print()  
s3, t3 = test_packages(ml_packages, '机器学习')
print()
s4, t4 = test_packages(dev_packages, '开发工具')

total_success = s1 + s2 + s3 + s4
total_packages = t1 + t2 + t3 + t4

print('=' * 40)
print(f'📊 总体成功率: {total_success}/{total_packages} ({total_success/total_packages*100:.1f}%)')

# GPU检查
try:
    import torch
    print(f'🔥 CUDA支持: {\"是\" if torch.cuda.is_available() else \"否\"}')
except:
    pass
"
}

# 主函数
main() {
    install_missing_packages
    install_torch_geometric
    handle_chemprop
    create_helper_scripts
    final_verification
    
    echo ""
    echo "🎉 KANO环境补充安装完成！"
    echo ""
    echo "📋 接下来你可以："
    echo "  测试环境:    bash test_env.sh"
    echo "  启动Jupyter: bash start_jupyter.sh"
    echo "  重新激活:    bash activate_kano.sh"
    echo ""
    echo "🔬 环境已就绪，可以开始研究了！"
}

# 执行
main "$@"
