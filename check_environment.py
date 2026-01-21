#!/usr/bin/env python3
"""
KANO环境检测脚本 - 检测GPU、依赖包和项目结构
"""
import sys
import subprocess
import importlib
import os
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    print(f"   当前版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 7:
        print("   ✅ Python版本符合要求 (≥3.7)")
        return True
    else:
        print("   ❌ Python版本不符合要求 (需要≥3.7)")
        return False

def check_cuda_gpu():
    """检查CUDA和GPU"""
    print("\n🚀 检查CUDA和GPU...")
    
    # 检查nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ NVCC可用")
            # 提取CUDA版本
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    print(f"   📦 {line.strip()}")
        else:
            print("   ❌ NVCC不可用")
    except FileNotFoundError:
        print("   ❌ NVCC未找到")
    
    # 检查PyTorch GPU支持
    try:
        import torch
        print(f"   🔥 PyTorch版本: {torch.__version__}")
        print(f"   🚀 CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   📦 PyTorch CUDA版本: {torch.version.cuda}")
            print(f"   💾 GPU设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   🖥️  GPU {i}: {name} ({memory:.1f} GB)")
        return torch.cuda.is_available()
    except ImportError:
        print("   ❌ PyTorch未安装")
        return False

def check_required_packages():
    """检查必需的Python包"""
    print("\n📦 检查必需的Python包...")
    
    required_packages = {
        'torch': '1.13.1',
        'rdkit': '2018.09.3', 
        'numpy': '1.20.3',
        'gensim': '4.2.0',
        'nltk': '3.4.5',
        'owl2vec_star': '0.2.1',
        'owlready2': '0.37',
        'torch_scatter': '2.0.9'
    }
    
    missing_packages = []
    installed_packages = {}
    
    for package, required_version in required_packages.items():
        try:
            # 特殊处理包名映射
            import_name = package
            if package == 'owl2vec_star':
                import_name = 'owl2vec_star'
            elif package == 'torch_scatter':
                import_name = 'torch_scatter'
            
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            installed_packages[package] = version
            print(f"   ✅ {package}: {version} (要求: {required_version})")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}: 未安装 (要求: {required_version})")
    
    return missing_packages, installed_packages

def check_project_structure():
    """检查项目目录结构"""
    print("\n📁 检查项目目录结构...")
    
    required_dirs = [
        'chemprop',
        'data', 
        'dumped',
        'initial',
        'KGembedding'
    ]
    
    required_files = [
        'pretrain.py',
        'train.py',
        'finetune.sh',
        'data/zinc15_250K.csv',
        'KGembedding/elementkg.owl'
    ]
    
    missing_items = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"   ✅ 目录存在: {dir_name}/")
        else:
            print(f"   ❌ 目录缺失: {dir_name}/")
            missing_items.append(dir_name)
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"   ✅ 文件存在: {file_name}")
        else:
            print(f"   ❌ 文件缺失: {file_name}")
            missing_items.append(file_name)
    
    return missing_items

def test_gpu_computation():
    """测试GPU计算能力"""
    print("\n⚡ 测试GPU计算能力...")
    
    try:
        import torch
        if torch.cuda.is_available():
            # 创建测试张量
            device = torch.device('cuda')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            # 计算时间
            import time
            torch.cuda.synchronize()
            start_time = time.time()
            
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            
            gpu_time = time.time() - start_time
            print(f"   ✅ GPU矩阵乘法测试成功!")
            print(f"   ⏱️  计算时间: {gpu_time:.4f}秒")
            print(f"   🎯 GPU已准备就绪，可加速分子模型训练!")
            return True
        else:
            print("   ❌ GPU不可用，将使用CPU模式")
            return False
    except Exception as e:
        print(f"   ❌ GPU测试失败: {str(e)}")
        return False

def main():
    """主检测函数"""
    print("=" * 60)
    print("🧬 KANO 环境检测工具")
    print("=" * 60)
    
    # 检查各项环境
    python_ok = check_python_version()
    gpu_ok = check_cuda_gpu()
    missing_packages, installed = check_required_packages()
    missing_structure = check_project_structure()
    
    if gpu_ok:
        gpu_compute_ok = test_gpu_computation()
    else:
        gpu_compute_ok = False
    
    # 总结报告
    print("\n" + "=" * 60)
    print("📋 检测结果总结")
    print("=" * 60)
    
    print(f"🐍 Python版本: {'✅ 通过' if python_ok else '❌ 不符合要求'}")
    print(f"🚀 GPU支持: {'✅ 可用' if gpu_ok else '❌ 不可用'}")
    print(f"⚡ GPU计算: {'✅ 正常' if gpu_compute_ok else '❌ 异常'}")
    print(f"📦 依赖包: {'✅ 完整' if not missing_packages else f'❌ 缺失{len(missing_packages)}个'}")
    print(f"📁 项目结构: {'✅ 完整' if not missing_structure else f'❌ 缺失{len(missing_structure)}项'}")
    
    # 建议
    print(f"\n💡 建议:")
    if missing_packages:
        print("   🔧 安装缺失的Python包:")
        for pkg in missing_packages:
            print(f"      pip install {pkg}")
    
    if not gpu_ok:
        print("   🚀 安装GPU支持以获得更好性能")
    
    if missing_structure:
        print("   📁 确保项目目录完整")
        for item in missing_structure:
            print(f"      缺失: {item}")
    
    # 运行建议
    print(f"\n🎯 下一步操作:")
    if python_ok and not missing_packages and not missing_structure:
        print("   ✅ 环境检测完成，可以开始运行KANO!")
        print("   🚀 建议运行顺序:")
        print("      1. bash finetune.sh  # 直接使用预训练模型")
        print("      2. python pretrain.py --exp_name 'pre-train' --exp_id 1 --step pretrain  # 重新预训练")
    else:
        print("   ⚠️  请先解决上述问题后再运行KANO")

if __name__ == "__main__":
    main()
