#!/bin/bash

echo "🧹 开始清理旧的KANO环境..."

# 如果使用conda环境
if command -v conda &> /dev/null; then
    echo "📦 检查conda环境..."
    
    # 列出所有conda环境
    echo "当前conda环境："
    conda env list
    
    # 删除kano相关环境（如果存在）
    read -p "是否删除conda环境 'kano'? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n kano -y 2>/dev/null || echo "kano环境不存在，跳过"
    fi
    
    # 删除其他可能的环境名
    for env_name in "KANO" "kano-env" "bioinfo"; do
        if conda env list | grep -q "$env_name"; then
            read -p "发现环境 '$env_name'，是否删除? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                conda env remove -n "$env_name" -y
            fi
        fi
    done
fi

# 清理pip缓存
echo "🗑️ 清理pip缓存..."
python3 -m pip cache purge 2>/dev/null || pip3 cache purge 2>/dev/null || echo "pip缓存清理完成"

# 清理Python缓存
echo "🗑️ 清理Python缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo "✅ 环境清理完成！"
echo ""
