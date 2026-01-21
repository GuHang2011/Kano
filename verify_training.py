import torch
import os

model_dir = "./dumped/bbbp_kapt_conservative/minimal_impact/"

print("=" * 80)
print("🔍 验证训练 Epochs")
print("=" * 80)

for i in range(5):
    model_path = os.path.join(model_dir, f'model_{i}', 'model.pt')
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 检查 checkpoint 中是否包含 epoch 信息
        if 'epoch' in checkpoint:
            print(f"\n✅ Model {i}: Trained to epoch {checkpoint['epoch']}")
        elif 'args' in checkpoint:
            print(f"\n✅ Model {i}: Checkpoint loaded (args found)")
            if hasattr(checkpoint['args'], 'epochs'):
                print(f"   Configured epochs: {checkpoint['args'].epochs}")
        else:
            print(f"\n⚠️ Model {i}: No epoch info in checkpoint")
            print(f"   Available keys: {list(checkpoint.keys())}")
    else:
        print(f"\n❌ Model {i}: NOT FOUND")

print("\n" + "=" * 80)
