import torch
import torch.nn as nn
import copy
import scipy.optimize
from utils import UnifiedModelManager

class FullWeightMatcher:
    def __init__(self, device='cuda'):
        self.device = device

    def get_permuted_param(self, param: torch.Tensor, perm: torch.Tensor, dim: int):
        """辅助函数：根据索引重排参数的指定维度"""
        return param.index_select(dim, perm)

    # =========================================================================
    # Step 1: 残差流正交对齐 (Global Orthogonal Alignment) [cite: 160-162, 191]
    # =========================================================================
    def procrustes_alignment(self, model_a: nn.Module, model_b: nn.Module):
        print(">>> [Step 1] Computing Global Orthogonal Alignment (Procrustes)...")
        
        weights_a = []
        weights_b = []
        
        # 收集所有"写入"残差流的权重 (Output Projections)
        # 这些矩阵的 Output Dimension 对应残差流的 Embedding Dimension
        for block_a, block_b in zip(model_a.visual.transformer.resblocks, 
                                    model_b.visual.transformer.resblocks):
            
            # Attn Output (c_proj/out_proj) [embed_dim, embed_dim]
            weights_a.append(block_a.attn.out_proj.weight.data)
            weights_b.append(block_b.attn.out_proj.weight.data)
            
            # MLP Output (c_proj) [embed_dim, mlp_dim]
            weights_a.append(block_a.mlp.c_proj.weight.data)
            weights_b.append(block_b.mlp.c_proj.weight.data)

        # 转换为 [N_samples, Embed_Dim] 形式
        # Linear.weight 是 [Out, In]，我们需要对齐 Dim 0
        W_A = torch.cat([w.t() for w in weights_a], dim=0) 
        W_B = torch.cat([w.t() for w in weights_b], dim=0)
        
        # 计算 SVD: M = B^T * A
        M = torch.matmul(W_B.t(), W_A)
        try:
            U, Sigma, Vh = torch.linalg.svd(M.float())
        except:
            M_cpu = M.cpu()
            U, Sigma, Vh = torch.linalg.svd(M_cpu)
            U, Vh = U.to(self.device), Vh.to(self.device)
            
        # 计算正交矩阵 O = U @ V^T
        O = torch.matmul(U, Vh).type(W_A.dtype) 
        print(f"    Computed Orthogonal Matrix O: {O.shape}")
        
        # --- 应用旋转 O 到 Model B ---
        with torch.no_grad():
            # 1. 旋转 Embeddings (残差流起点)
            model_b.visual.class_embedding.data = torch.matmul(O, model_b.visual.class_embedding.data.unsqueeze(1)).squeeze(1)
            model_b.visual.positional_embedding.data = torch.matmul(model_b.visual.positional_embedding.data, O.t())
            
            # Patch Embedding: Conv2d [embed, 3, k, k] -> 需要旋转 Out Channels (Dim 0)
            conv_w = model_b.visual.conv1.weight.data
            out_ch, in_ch, k1, k2 = conv_w.shape
            conv_w_flat = conv_w.view(out_ch, -1) 
            conv_w_rotated = torch.matmul(O, conv_w_flat).view(out_ch, in_ch, k1, k2)
            model_b.visual.conv1.weight.data.copy_(conv_w_rotated)

            # 2. 旋转 Transformer Blocks
            for block in model_b.visual.transformer.resblocks:
                # 输入端 (Input Projections): 右乘 O.T (x -> Ox)
                block.ln_1.weight.data = torch.matmul(O, block.ln_1.weight.data)
                block.ln_1.bias.data = torch.matmul(O, block.ln_1.bias.data)
                block.attn.in_proj_weight.data = torch.matmul(block.attn.in_proj_weight.data, O.t()) # Linear In
                # in_proj_bias 不需要变，因为它加在 QKV 输出上，不直接受输入旋转影响

                block.ln_2.weight.data = torch.matmul(O, block.ln_2.weight.data)
                block.ln_2.bias.data = torch.matmul(O, block.ln_2.bias.data)
                block.mlp.c_fc.weight.data = torch.matmul(block.mlp.c_fc.weight.data, O.t()) # Linear In
                
                # 输出端 (Output Projections): 左乘 O (y -> Oy)
                block.attn.out_proj.weight.data = torch.matmul(O, block.attn.out_proj.weight.data)
                block.attn.out_proj.bias.data = torch.matmul(O, block.attn.out_proj.bias.data)

                block.mlp.c_proj.weight.data = torch.matmul(O, block.mlp.c_proj.weight.data)
                block.mlp.c_proj.bias.data = torch.matmul(O, block.mlp.c_proj.bias.data)
                
            # 3. 旋转 Final Projection
            model_b.visual.ln_post.weight.data = torch.matmul(O, model_b.visual.ln_post.weight.data)
            model_b.visual.ln_post.bias.data = torch.matmul(O, model_b.visual.ln_post.bias.data)
            model_b.visual.proj.data = torch.matmul(O, model_b.visual.proj.data)
            
        print("    Global Orthogonal Alignment Done.")

    # =========================================================================
    # Step 2: Attention 层对齐 (Head Alignment) 
    # =========================================================================
    def match_attention_heads(self, model_a: nn.Module, model_b: nn.Module):
        print(">>> [Step 2] Matching Attention Heads...")
        
        with torch.no_grad():
            for layer_idx, (block_a, block_b) in enumerate(zip(model_a.visual.transformer.resblocks, 
                                                               model_b.visual.transformer.resblocks)):
                # 获取配置
                embed_dim = block_a.attn.out_proj.in_features
                num_heads = block_a.attn.num_heads
                head_dim = embed_dim // num_heads
                
                # --- 1. 提取 Q, K, V, O 权重 ---
                # OpenCLIP in_proj_weight 形状通常是 [3*embed, embed]
                # 顺序通常是 [Q_all, K_all, V_all] 或交错。OpenCLIP 是 concat([q,k,v])
                
                # 提取 Model A
                qkv_a = block_a.attn.in_proj_weight.data # [3*D, D]
                q_a, k_a, v_a = qkv_a.chunk(3, dim=0)    # Split dim 0 -> [D, D]
                o_a = block_a.attn.out_proj.weight.data  # [D, D] (Rows=Embed, Cols=Heads_Concat)
                
                # 提取 Model B
                qkv_b = block_b.attn.in_proj_weight.data
                q_b, k_b, v_b = qkv_b.chunk(3, dim=0)
                o_b = block_b.attn.out_proj.weight.data

                # --- 2. 重塑为 Heads ---
                # Input Proj (Q, K, V): [Num_Heads, Head_Dim, Embed_Dim]
                # 对 Dim 0 (Output features) 进行 reshape
                Wq_a = q_a.view(num_heads, head_dim, embed_dim)
                Wk_a = k_a.view(num_heads, head_dim, embed_dim)
                Wv_a = v_a.view(num_heads, head_dim, embed_dim)
                
                Wq_b = q_b.view(num_heads, head_dim, embed_dim)
                Wk_b = k_b.view(num_heads, head_dim, embed_dim)
                Wv_b = v_b.view(num_heads, head_dim, embed_dim)
                
                # Output Proj (O): [Embed_Dim, Num_Heads, Head_Dim]
                # 输入维度 (Dim 1) 对应 Heads
                Wo_a = o_a.view(embed_dim, num_heads, head_dim).permute(1, 0, 2) # -> [Num_Heads, Embed, Head_Dim]
                Wo_b = o_b.view(embed_dim, num_heads, head_dim).permute(1, 0, 2)

                # --- 3. 计算电路矩阵 (Circuit Matrices) [cite: 184-186] ---
                # QK Circuit: Wq @ Wk.T -> [Num_Heads, Head_Dim, Head_Dim] (简化版，仅比较功能)
                # OV Circuit: Wv @ Wo   -> [Num_Heads, Head_Dim, Embed] @ [Num_Heads, Embed, Head_Dim] ?
                # 论文定义的 OV 是 W_V * W_O。
                # 注意矩阵乘法维度匹配：
                # W_Q: [Head_Dim, Embed], W_K^T: [Embed, Head_Dim] -> QK: [Head_Dim, Head_Dim]
                # W_V: [Head_Dim, Embed], W_O: [Embed, Head_Dim]   -> OV: [Head_Dim, Head_Dim] (In output space)
                
                # 这里的张量是 [Heads, Head_Dim, Embed]。
                # QK_i = Wq_i @ Wk_i.T
                QK_a = torch.matmul(Wq_a, Wk_a.transpose(1, 2)) # [H, Head_Dim, Head_Dim]
                QK_b = torch.matmul(Wq_b, Wk_b.transpose(1, 2))
                
                # OV_i = Wo_i @ Wv_i (注意顺序，Wo将Head_Dim映射回Embed，Wv将Embed映射到Head_Dim)
                # 论文公式: OV_i = W_O * W_V (全局视角)。
                # 这里 Wo_a 是 [H, Embed, Head_Dim]。Wv_a 是 [H, Head_Dim, Embed]。
                # 乘积应为 [H, Embed, Embed]。
                OV_a = torch.matmul(Wo_a, Wv_a) # [H, Embed, Embed]
                OV_b = torch.matmul(Wo_b, Wv_b)

                # --- 4. 构建 Cost Matrix ---
                # Cost[i, j] = ||QK_a[i] - QK_b[j]|| + ||OV_a[i] - OV_b[j]||
                # 展平以便计算范数
                QK_a_flat = QK_a.view(num_heads, -1)
                QK_b_flat = QK_b.view(num_heads, -1)
                OV_a_flat = OV_a.view(num_heads, -1)
                OV_b_flat = OV_b.view(num_heads, -1)
                
                # 计算两两距离 (利用广播或cdist)
                dist_qk = torch.cdist(QK_a_flat, QK_b_flat, p=2) ** 2
                dist_ov = torch.cdist(OV_a_flat, OV_b_flat, p=2) ** 2
                cost_matrix = (dist_qk + dist_ov).cpu().numpy()
                
                # --- 5. 求解最佳匹配 ---
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
                # col_ind[i] 表示 Model B 的第 col_ind[i] 个头应该放到位置 i
                perm = torch.tensor(col_ind, device=self.device)
                
                # --- 6. 应用置换到 Model B ---
                # 我们需要按照 `perm` 重排 Model B 的 Heads
                # 意味着：现在的第 i 个位置，应该放原来的第 perm[i] 个头
                
                # 重排 Q, K, V (Input Proj)
                # Wq_b: [H, Hd, D] -> index_select dim 0
                q_b_new = Wq_b.index_select(0, perm).reshape(embed_dim, embed_dim) # Flatten back
                k_b_new = Wk_b.index_select(0, perm).reshape(embed_dim, embed_dim)
                v_b_new = Wv_b.index_select(0, perm).reshape(embed_dim, embed_dim)
                
                # 重新拼接并赋值给 in_proj_weight
                block_b.attn.in_proj_weight.data = torch.cat([q_b_new, k_b_new, v_b_new], dim=0)
                
                # 重排 Bias (in_proj_bias)
                if block_b.attn.in_proj_bias is not None:
                    bias_q, bias_k, bias_v = block_b.attn.in_proj_bias.data.chunk(3, dim=0)
                    bias_q = bias_q.view(num_heads, head_dim).index_select(0, perm).reshape(embed_dim)
                    bias_k = bias_k.view(num_heads, head_dim).index_select(0, perm).reshape(embed_dim)
                    bias_v = bias_v.view(num_heads, head_dim).index_select(0, perm).reshape(embed_dim)
                    block_b.attn.in_proj_bias.data = torch.cat([bias_q, bias_k, bias_v], dim=0)

                # 重排 Output Proj (O)
                # Wo_b: [H, D, Hd] -> index_select dim 0
                # 需要还原回 [D, H*Hd] (Linear weight shape: [Out, In])
                o_b_permuted = Wo_b.index_select(0, perm).permute(1, 0, 2).reshape(embed_dim, embed_dim)
                block_b.attn.out_proj.weight.data = o_b_permuted
                
        print("    Attention Head Alignment Done.")

    # =========================================================================
    # Step 3: MLP (FFN) 层对齐 (Neuron Permutation) 
    # =========================================================================
    def match_mlps(self, model_a: nn.Module, model_b: nn.Module):
        print(">>> [Step 3] Matching MLP Neurons (Permutation)...")
        
        with torch.no_grad():
            for i, (block_a, block_b) in enumerate(zip(model_a.visual.transformer.resblocks, 
                                                       model_b.visual.transformer.resblocks)):
                w1_a = block_a.mlp.c_fc.weight.data
                w1_b = block_b.mlp.c_fc.weight.data
                w2_a = block_a.mlp.c_proj.weight.data
                w2_b = block_b.mlp.c_proj.weight.data
                
                # Normalize
                w1_a_n = w1_a / w1_a.norm(dim=1, keepdim=True)
                w1_b_n = w1_b / w1_b.norm(dim=1, keepdim=True)
                w2_a_n = w2_a / w2_a.norm(dim=0, keepdim=True) 
                w2_b_n = w2_b / w2_b.norm(dim=0, keepdim=True)
                
                # Similarity [Hidden, Hidden]
                sim_matrix = torch.matmul(w1_a_n, w1_b_n.t()) + torch.matmul(w2_a_n.t(), w2_b_n)
                cost_matrix = -sim_matrix.cpu().numpy()
                
                # Solve Assignment
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
                perm = torch.tensor(col_ind, device=self.device)
                
                # Apply Permutation to Model B
                # 1. c_fc Output (rows)
                block_b.mlp.c_fc.weight.data = self.get_permuted_param(block_b.mlp.c_fc.weight.data, perm, 0)
                block_b.mlp.c_fc.bias.data = self.get_permuted_param(block_b.mlp.c_fc.bias.data, perm, 0)
                
                # 2. c_proj Input (cols)
                block_b.mlp.c_proj.weight.data = self.get_permuted_param(block_b.mlp.c_proj.weight.data, perm, 1)
                
        print("    MLP Permutation Done.")

    # =========================================================================
    # 主融合函数
    # =========================================================================
    def fuse_models(self, manager: UnifiedModelManager, 
                    model_name_a: str, 
                    model_name_b: str, 
                    alpha: float = 0.5):
        
        print(f"Loading models for Full Weight Matching Fusion (Alpha={alpha})...")
        model_a = manager.get_model(model_name_a)
        model_b = manager.get_model(model_name_b)
        
        # 深拷贝 Model B 用于对齐
        model_b_aligned = copy.deepcopy(model_b)
        
        # 执行三步对齐
        # 1. 残差流全局旋转
        self.procrustes_alignment(model_a, model_b_aligned)
        
        # 2. Attention Heads 局部置换
        self.match_attention_heads(model_a, model_b_aligned)
        
        # 3. MLP Neurons 局部置换
        self.match_mlps(model_a, model_b_aligned)
        
        # 4. 线性插值
        print(f"Interpolating weights...")
        fusion_model = copy.deepcopy(model_a)
        
        for param_f, param_a, param_b_aln in zip(fusion_model.parameters(), 
                                                 model_a.parameters(), 
                                                 model_b_aligned.parameters()):
            param_f.data = (1.0 - alpha) * param_a.data + alpha * param_b_aln.data
            
        return fusion_model
