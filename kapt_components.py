# ===== kapt_components.py =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class DynamicPromptPool(nn.Module):
    """动态提示池（DPP）"""

    def __init__(self, num_tasks, prompt_dim, num_tokens=5):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(num_tasks, num_tokens, prompt_dim))
        self.attention = nn.MultiheadAttention(prompt_dim, num_heads=4, batch_first=True)

    def forward(self, task_id, batch_size):
        task_prompt = self.prompts[task_id].unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attention(task_prompt, task_prompt, task_prompt)
        return attended.mean(dim=1)  # [batch_size, prompt_dim]


class StructureAwarePromptGenerator(nn.Module):
    """结构感知提示生成器（SPG）"""

    def __init__(self, node_dim, hidden_dim, prompt_dim):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim)
        )
        self.graph_pooling = global_mean_pool

    def forward(self, batch):
        node_feats = self.node_encoder(batch.x.float())
        graph_prompt = self.graph_pooling(node_feats, batch.batch)
        return graph_prompt  # [batch_size, prompt_dim]


class HierarchicalPromptAggregator(nn.Module):
    """层次化提示聚合器（HPA）"""

    def __init__(self, prompt_dim, hidden_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(prompt_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim)
        )

    def forward(self, task_prompt, struct_prompt):
        combined = torch.cat([task_prompt, struct_prompt], dim=-1)
        return self.fusion(combined)  # [batch_size, prompt_dim]


class NodeLevelPromptRefiner(nn.Module):
    """节点级提示细化器（NLPR）"""

    def __init__(self, node_dim, prompt_dim):
        super().__init__()
        self.refiner = nn.Sequential(
            nn.Linear(node_dim + prompt_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, node_feats, graph_prompt, batch_idx):
        prompt_expanded = graph_prompt[batch_idx]
        combined = torch.cat([node_feats, prompt_expanded], dim=-1)
        return self.refiner(combined)


class KAPTPromptModule(nn.Module):
    """完整的KAPT提示模块"""

    def __init__(self, num_tasks, node_dim, hidden_dim, prompt_dim):
        super().__init__()
        self.dpp = DynamicPromptPool(num_tasks, prompt_dim)
        self.spg = StructureAwarePromptGenerator(node_dim, hidden_dim, prompt_dim)
        self.hpa = HierarchicalPromptAggregator(prompt_dim, hidden_dim)
        self.nlpr = NodeLevelPromptRefiner(hidden_dim, prompt_dim)

    def forward(self, batch, task_id):
        batch_size = batch.batch.max().item() + 1
        task_prompt = self.dpp(task_id, batch_size)
        struct_prompt = self.spg(batch)
        fused_prompt = self.hpa(task_prompt, struct_prompt)
        return fused_prompt


class PromptInjector(nn.Module):
    """提示注入器"""

    def __init__(self, hidden_dim, prompt_dim):
        super().__init__()
        self.proj = nn.Linear(prompt_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, prompt, batch_idx):
        prompt_expanded = prompt[batch_idx]
        prompt_proj = self.proj(prompt_expanded)
        gate_input = torch.cat([x, prompt_proj], dim=-1)
        gate_weight = self.gate(gate_input)
        return x + gate_weight * prompt_proj
