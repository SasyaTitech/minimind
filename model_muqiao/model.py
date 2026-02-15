from transformers import PretrainedConfig
import torch
import torch.nn as nn

# PretrainedConfig是Hugging Face的模型配置基类，训练发布的时候会生成config.json
# 别人from_pretrained就会先读配置然后构建模型
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )



# 1. RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-5):
        super().__init__() 
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) #需要被学习的的得用这个包裹

    def _norm(self, x:torch.Tensor): #首字母大写通常表示类/类型（class/type），小写通常是函数，变量
        return torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps) * x #求平均会导致维度退化
        # 计算的时候把x升到f32计算比较稳
    def forward(self, x:torch.Tensor):
        return self._norm(x) * self.weight.type_as(x) #本身是float32的weight可能会改掉整体的类型
        #比如这里如果x是float16或者bfloat16可能会被强行拉到float32
    

# 2. RoPE 相对位置编码



