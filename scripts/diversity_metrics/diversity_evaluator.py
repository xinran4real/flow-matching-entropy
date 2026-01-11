"""
Advanced Diversity Evaluator for Image Generation
集成 Vendi Score, Conditional Vendi, RKE, PRDC 等指标
使用 Accelerator API 的分布式 Diversity 评估
"""

import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import warnings
from torch.utils.data import Dataset

from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from sklearn.metrics.pairwise import euclidean_distances


class DiversityEvaluator:
    """
    高级多样性评估器，使用CLIP
    """
    def __init__(self, device='cuda'):
        self.device = device
        self._init_clip()
        self.rke_evaluator = None
        print(f"DiversityEvaluator initialized with CLIP")
    
    def _init_clip(self):
        """初始化CLIP作为特征提取器"""
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
            self.clip_model.eval()
            print("✓ Using CLIP-ViT-L/14 for feature extraction")
        except Exception as e:
            raise RuntimeError(f"Could not initialize CLIP: {e}")
    
    def extract_features(self, images):
        """
        提取图像特征（使用CLIP）
        Args:
            images: torch.Tensor [N, C, H, W] in range [0, 1]
        Returns:
            features: torch.Tensor [N, feature_dim] (always float32)
        """
        # CLIP预处理：resize到224x224
        images = torch.nn.functional.interpolate(
            images, size=(224, 224), mode='bilinear', align_corners=False
        )
        
        # 标准化 (CLIP的normalization)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        
        # 提取特征
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
            # L2 normalize
            features = features / features.norm(dim=-1, keepdim=True)
        
        # 确保返回 float32
        return features.float()
    
    def compute_vendi_score(self, features_np, q=1):
        """
        Vendi Score: q=1 (Shannon Entropy)
        Kernel: Linear/Cosine (因为 features 已经 normalize 了，XX^T 就是 Cosine)
        """
        # features_np shape: [N, D]
        # X @ X.T -> [N, N] Cosine Similarity Matrix
        K = features_np @ features_np.T 
        # 归一化核矩阵，使得 trace(K) = 1 (也就是除以 N)
        n = K.shape[0]
        K_norm = K / n 
        
        # 计算特征值
        # 使用 eigh 用于对称矩阵，比 eig 更快更稳
        w = np.linalg.eigvalsh(K_norm)
        
        # 过滤掉数值误差导致的微小负值
        w = w[w > 1e-12]
        
        # Shannon Entropy: -sum(lambda * log(lambda))
        entropy = -np.sum(w * np.log(w))
        
        return np.exp(entropy)

    def compute_rke(self, features_np, sigma='auto'):
        """
        RKE-MC (Mode Count): q=2 (Renyi Entropy)
        Kernel: Gaussian (RBF)
        Formula: exp(H2) = 1 / sum(lambda^2) = 1 / Trace(K_norm @ K_norm)
        """
        import scipy.spatial.distance as dist
        
        # 1. 计算成对欧氏距离平方
        # pdist 返回压缩的距离矩阵，squareform 转为 N*N
        dists_sq = dist.squareform(dist.pdist(features_np, 'sqeuclidean'))
        
        # 2. 确定 Bandwidth (sigma)
        if sigma == 'auto':
            # 论文中常用的启发式：取距离的中位数
            # 注意：pdist 返回的是距离，不是平方，所以这里要开根号取中位数，或者直接在平方上操作
            median_dist_sq = np.median(dists_sq)
            # 防止 sigma 为 0
            sigma_sq = median_dist_sq if median_dist_sq > 0 else 1.0
        else:
            sigma_sq = sigma ** 2

        # 3. 构建高斯核矩阵 K_ij = exp(- ||x-y||^2 / (2*sigma^2))
        # 注意：Jalali 论文中有时用 sigma^2 有时用 2*sigma^2，通常 RBF 是 2*sigma^2
        # 我们这里采用标准 RBF 定义
        K = np.exp(-dists_sq / (2 * sigma_sq))
        
        # 4. 归一化核矩阵
        n = K.shape[0]
        K_norm = K / n
        
        # 5. 计算 RKE-MC (q=2)
        # 公式推导：
        # H2 = -log(sum(lambda_i^2)) = -log(Trace(K_norm^2))
        # RKE-MC = exp(H2) = 1 / Trace(K_norm^2)
        # Trace(A @ A) = sum(A_ij * A_ji) = sum(A_ij^2) (因为 A 对称)
        
        trace_k_squared = np.sum(K_norm ** 2)
        rke_mc = 1.0 / trace_k_squared
        
        return rke_mc


def evaluate_diversity_distributed_accelerator(
    pipeline,
    test_dataloader,
    num_seeds,
    accelerator,
    resolution,
    num_steps,
    guidance_scale,
    text_encoders,
    tokenizers,
    evaluator=None,
):
    """
    使用 Accelerator API 的分布式 diversity 评估

    Args:
        pipeline: StableDiffusion3Pipeline
        test_dataloader: torch.utils.data.DataLoader (Must be prepared by accelerator)
        num_seeds: int
        accelerator: Accelerator instance
        resolution: int
        num_steps: int
        guidance_scale: float
        text_encoders: list of text encoders
        tokenizers: list of tokenizers
        evaluator: DiversityEvaluator instance
    
    Returns:
        diversity_metrics: dict (只在主进程返回有效值)
        images_by_prompt: dict (每个进程返回自己生成的图像)
    """
    # 初始化 evaluator (每个进程都需要)
    if evaluator is None:
        evaluator = DiversityEvaluator(device=accelerator.device)
    
    pipeline.transformer.eval()
    device = accelerator.device

    # 用于存储当前进程计算出的所有特征
    local_features_list = []
    # ============ Step 1: 遍历 DataLoader (自动处理分布式分发) ============
    # 注意：test_dataloader 必须在传入前已经在 train_sd3.py 中经过 accelerator.prepare()
    
    # 进度条只在每个机器的主进程显示
    disable_tqdm = not accelerator.is_local_main_process
    
    # 根据 train_sd3.py 的 collate_fn，batch 是一个 tuple: (prompts_list, metadatas_list)
    for batch_prompts, batch_metadatas in tqdm(
        test_dataloader, 
        desc="Diversity Sampling", 
        disable=disable_tqdm
    ):
        
        # 遍历当前 batch 中的每一个 prompt
        for prompt in batch_prompts:
            # 编码 prompt (这对 SD3 是必要的预处理)
            # 注意：encode_prompt 函数内部可能不带 no_grad，这里外层加一个保险
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders, tokenizers, [prompt], max_sequence_length=128
                )
                prompt_embeds = prompt_embeds.to(device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            
            # 对每个 prompt 生成 num_seeds 张图片
            for seed in range(num_seeds):
                generator = torch.Generator(device=device).manual_seed(seed)
                
                with torch.no_grad():
                    # 生成图像
                    output = pipeline(
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        height=resolution,
                        width=resolution,
                        generator=generator,
                        output_type="pt", # 返回 tensor [1, C, H, W]
                    ).images
                    
                    # 提取特征
                    # output 是 [1, 3, H, W]，evaluator 需要处理这个维度
                    features = evaluator.extract_features(output) # 返回 [1, feature_dim]
                    
                    # 显式转换为 float32 并移除 batch 维度存入列表
                    local_features_list.append(features[0].float())
            
            # 显式清理缓存，防止长序列生成时 OOM
            torch.cuda.empty_cache()
    
    # ============ Step 2: 堆叠本地特征 ============
    if len(local_features_list) > 0:
        local_features = torch.stack(local_features_list).to(device) # [Local_N, D]
    else:
        # 处理极其罕见的空数据情况（例如某些 GPU 分不到数据）
        # 创建一个空的 tensor，维度需要匹配以便 gather
        dummy_dim = evaluator.clip_model.visual.output_dim if hasattr(evaluator, 'clip_model') else 768
        local_features = torch.zeros((0, dummy_dim), device=device, dtype=torch.float32)

    if accelerator.is_local_main_process:
        print(f"[Process {accelerator.process_index}] Computed features shape: {local_features.shape}")

    # ============ Step 3: 收集所有 GPU 的特征 ============
    accelerator.wait_for_everyone()
    
    # gather 将所有进程的 tensor 在第 0 维拼接
    # 假如 GPU 0 有 100 个，GPU 1 有 100 个，结果就是 200 个
    # 注意：如果 DataLoader 有 padding (drop_last=False)，这里可能会有少量的重复样本
    # 对于 Diversity 指标，少量重复样本是可以接受的，或者使用 gather_for_metrics (需新版 accelerate)
    try:
        all_features_gathered = accelerator.gather(local_features)
    except Exception as e:
        # 容错处理：如果 tensor 大小不一致导致 gather 失败（通常 accelerator 会自动 pad，但以防万一）
        if accelerator.is_main_process:
            print(f"Warning: Gather failed ({e}), falling back to local evaluation (inaccurate).")
        all_features_gathered = local_features

    # ============ Step 4: 主进程计算指标 ============
    diversity_metrics = {}
    
    if accelerator.is_main_process:
        # 转移到 CPU 计算以节省显存，并转为 numpy
        all_features_np = all_features_gathered.cpu().numpy()
        total_samples = all_features_np.shape[0]
        
        print(f"\n{'='*70}")
        print(f"[Main Process] Computing diversity metrics on {total_samples} images...")
        print(f"{'='*70}")
        
        # 1. Vendi Score (q=1, Linear/Cosine Kernel)
        # 确保 DiversityEvaluator 中已更新为 compute_vendi_score
        if hasattr(evaluator, 'compute_vendi_score'):
            try:
                print("[Rank 0] Computing Vendi Score...")
                vendi_score = evaluator.compute_vendi_score(all_features_np, q=1)
                diversity_metrics['vendi_score'] = vendi_score
                print(f"[Rank 0]   ✓ Vendi Score: {vendi_score:.4f}")
            except Exception as e:
                print(f"[Rank 0]   x Vendi Score failed: {e}")

        # 2. RKE-MC (q=2, Gaussian Kernel)
        # 确保 DiversityEvaluator 中已更新为 compute_rke (使用精确矩阵计算)
        if hasattr(evaluator, 'compute_rke'):
            try:
                print("[Rank 0] Computing RKE (Gaussian Kernel)...")
                # 使用 'auto' 让算法基于距离中位数自动选择 bandwidth
                rke_score = evaluator.compute_rke(all_features_np, sigma='auto')
                diversity_metrics['rke'] = rke_score
                print(f"[Rank 0]   ✓ RKE: {rke_score:.4f}")
            except Exception as e:
                print(f"[Rank 0]   x RKE failed: {e}")

        print(f"{'='*70}\n")

    # 再次同步，确保主进程计算完之前其他进程不会提前退出或进入下一轮
    accelerator.wait_for_everyone()
    return diversity_metrics, {} # images_by_prompt


def run_diversity_evaluation(
    pipeline,
    test_dataloader,
    num_seeds,
    accelerator,
    resolution,
    num_steps,
    guidance_scale,
    text_encoders,
    tokenizers,
    evaluator=None,
):
    """
    便捷包装函数
    """
    # 初始化 evaluator
    if evaluator is None:
        evaluator = DiversityEvaluator(device=accelerator.device)
    
    # 全局 diversity 评估
    global_metrics, images_by_prompt = evaluate_diversity_distributed_accelerator(
        pipeline=pipeline,
        test_prompts=test_dataloader,
        num_seeds=num_seeds,
        accelerator=accelerator,
        resolution=resolution,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        evaluator=evaluator,
    )
    
    return global_metrics