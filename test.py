import torch
from torch_scatter import scatter_add

# 确认CUDA可用
print("CUDA是否可用：", torch.cuda.is_available())
print("显卡名称：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无CUDA显卡")

# 使用CUDA测试
if torch.cuda.is_available():
    src = torch.tensor([[1, 2], [3, 4], [5, 6]]).cuda()
    index = torch.tensor([0, 0, 1]).cuda()
    out = scatter_add(src, index, dim=0)
    print("CUDA测试输出：", out.cpu())  # 转回CPU打印，输出tensor([[4, 6], [5, 6]])
else:
    print("未检测到CUDA，使用CPU运行")
    src = torch.tensor([[1, 2], [3, 4], [5, 6]])
    index = torch.tensor([0, 0, 1])
    out = scatter_add(src, index, dim=0)
    print("CPU测试输出：", out)