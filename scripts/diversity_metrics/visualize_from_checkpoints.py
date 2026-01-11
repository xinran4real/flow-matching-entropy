"""
从训练checkpoint加载模型并可视化去噪过程
使用方法: python visualize_from_checkpoints.py --config config/your_config.py
"""

from collections import defaultdict
import os
import glob
import re
from absl import app, flags
from ml_collections import config_flags
import torch
import wandb
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel, LoraConfig
import tempfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import hashlib

from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from scripts.train_sd3_nsr import TextPromptDataset

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", "/workspace/flow_grpo_2/logs/pickscore/sd3.5-M/eval_ode_log_prob/checkpoints", 
                    "Directory containing checkpoints")
flags.DEFINE_string("checkpoint_pattern", "checkpoint-*", 
                    "Pattern to match checkpoint directories")
flags.DEFINE_string("wandb_project", "flow_grpo", "WandB project name")
flags.DEFINE_string("wandb_run_name", "checkpoint_visualization", "WandB run name")
flags.DEFINE_bool("visualize_all", False, "Visualize all checkpoints or only specific ones")
flags.DEFINE_list("checkpoint_steps", None, "Specific checkpoint steps to visualize (e.g., 100,200,500)")

# 全局字典，用于存储历史数据（键为prompt的hash）
# 添加前缀以区分checkpoint加载的结果
_checkpoint_visualization_history = defaultdict(lambda: {
    'checkpoint_steps': [],
    'ode_log_probs_per_step': [],
    'divergences_per_step': []
})


def visualize_latents(latents, vae):
    """将latents转换为可视化的图像"""
    with torch.no_grad():
        latents_scaled = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        latents_scaled = latents_scaled.to(dtype=vae.dtype)
        images = vae.decode(latents_scaled, return_dict=False)[0]
        images = (images + 1) / 2.0 if images.min() < 0 else images
        images = images.clamp(0, 1)
    return images


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    """计算文本嵌入"""
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def load_pipeline_with_checkpoint(checkpoint_path, base_model_path, device, pipeline_cache=None):
    """
    加载基础模型并应用checkpoint的LoRA权重
    
    Args:
        checkpoint_path: checkpoint目录路径
        base_model_path: 基础模型路径
        device: 设备
        pipeline_cache: 可选的缓存pipeline（用于重用基础模型）
    
    Returns:
        pipeline: 加载了checkpoint的pipeline
    """
    # 如果提供了缓存的pipeline，只需要替换LoRA权重
    if pipeline_cache is not None:
        print(f"Reusing cached pipeline")
        pipeline = pipeline_cache
        
        # 如果之前有LoRA adapter，先移除
        if hasattr(pipeline.transformer, 'peft_config'):
            # 卸载之前的adapter
            try:
                pipeline.transformer = pipeline.transformer.unload()
            except:
                pass
    else:
        print(f"Loading base model from {base_model_path}")
        pipeline = StableDiffusion3Pipeline.from_pretrained(base_model_path)
        
        # 冻结不需要训练的模型
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)
        pipeline.text_encoder_3.requires_grad_(False)
        
        # 移动到设备
        pipeline.vae.to(device, dtype=torch.float32)
        pipeline.text_encoder.to(device)
        pipeline.text_encoder_2.to(device)
        pipeline.text_encoder_3.to(device)
        pipeline.transformer.to(device)
        
        pipeline.safety_checker = None
        pipeline.set_progress_bar_config(disable=True)
    
    # 加载LoRA权重
    lora_path = os.path.join(checkpoint_path, "lora")
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}")
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer, 
            lora_path,
            is_trainable=False  # 推理模式
        )
        pipeline.transformer.set_adapter("default")
    else:
        print(f"Warning: LoRA path not found at {lora_path}")
    
    return pipeline


def visualize_denoising_from_checkpoint(
    pipeline, prompt, text_encoders, tokenizers, config, device,
    checkpoint_step, num_visualize_steps=8, prompt_name=""
):
    """
    从checkpoint加载的模型可视化去噪过程
    注意：使用不同的wandb key前缀避免覆盖训练时的结果
    """
    pipeline.transformer.eval()
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    
    prompt_list = [prompt] if isinstance(prompt, str) else [prompt[0]]
    
    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
        prompt_list, text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(1, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(1, 1)
    
    with torch.no_grad():
        images, all_latents, _, _, all_ode_log_probs, all_divergences, ode_log_prob_final = pipeline_with_logprob(
            pipeline,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=sample_neg_prompt_embeds,
            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
            num_inference_steps=config.sample.eval_num_steps,
            guidance_scale=config.sample.guidance_scale,
            output_type="pt",
            height=config.resolution,
            width=config.resolution,
            noise_level=0.0,
            eval_log_prob=True,
            visualize_steps=True,
        )
    
    # 立即将大张量移到CPU以释放GPU内存
    images = images.cpu()
    all_latents = [lat.cpu() for lat in all_latents]
    all_ode_log_probs = [olp.cpu() for olp in all_ode_log_probs]
    all_divergences = [div.cpu() for div in all_divergences]
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    # 提取数据
    divergences_np = torch.stack(all_divergences).numpy()
    ode_log_probs_np = torch.stack(all_ode_log_probs).numpy()
    actual_num_steps = len(divergences_np)
    
    # 存储历史数据（使用checkpoint_step而不是global_step）
    prompt_hash = hashlib.md5(prompt_list[0].encode()).hexdigest()[:8]
    history_key = f"{prompt_name}_{prompt_hash}" if prompt_name else prompt_hash
    
    _checkpoint_visualization_history[history_key]['checkpoint_steps'].append(checkpoint_step)
    _checkpoint_visualization_history[history_key]['ode_log_probs_per_step'].append(ode_log_probs_np)
    _checkpoint_visualization_history[history_key]['divergences_per_step'].append(divergences_np)
    
    # 选择要可视化的步骤
    total_steps = len(all_latents) - 1
    if total_steps <= num_visualize_steps:
        step_indices = list(range(total_steps))
    else:
        step_indices = [int(i * total_steps / (num_visualize_steps - 1)) for i in range(num_visualize_steps - 1)]
        step_indices.append(total_steps - 1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wandb_images = []
        
        # 可视化中间步骤
        for vis_idx, step_idx in enumerate(step_indices):
            latent = all_latents[step_idx].to(device)  # 临时移到GPU进行解码
            vis_image = visualize_latents(latent, pipeline.vae)
            vis_image = vis_image.cpu()  # 立即移回CPU
            latent = latent.cpu()  # 释放GPU内存
            torch.cuda.empty_cache()
            
            vis_image_np = (vis_image[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pil = Image.fromarray(vis_image_np)
            pil = pil.resize((config.resolution, config.resolution))
            pil.save(os.path.join(tmpdir, f"step_{vis_idx}.jpg"))
            
            if step_idx > 0:
                ode_log_prob = all_ode_log_probs[step_idx - 1][0].item()
                divergence = all_divergences[step_idx - 1][0].item()
                caption = f"Step {step_idx}/{total_steps}\nODE_log_prob: {ode_log_prob:.4f}\nDivergence: {divergence:.4f}"
            else:
                caption = f"Step {step_idx}/{total_steps} (Initial Noise)"
            
            wandb_images.append(wandb.Image(os.path.join(tmpdir, f"step_{vis_idx}.jpg"), caption=caption))
        
        # 最终图像
        final_image_np = (images[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil = Image.fromarray(final_image_np)
        pil = pil.resize((config.resolution, config.resolution))
        pil.save(os.path.join(tmpdir, "final.jpg"))
        wandb_images.append(wandb.Image(os.path.join(tmpdir, "final.jpg"), 
                                       caption=f"Final Image\nPrompt: {prompt_list[0][:100]}"))
        
        # === 方案A: 叠加历史曲线 ===
        history = _checkpoint_visualization_history[history_key]
        num_history = len(history['checkpoint_steps'])
        
        if num_history > 1:
            fig_div, ax_div = plt.subplots(figsize=(10, 6))
            fig_ode, ax_ode = plt.subplots(figsize=(10, 6))
            
            cmap = plt.cm.viridis
            colors = [cmap(i / (num_history - 1)) for i in range(num_history)]
            
            for idx, (ckpt_step, div_curve, ode_curve) in enumerate(zip(
                history['checkpoint_steps'],
                history['divergences_per_step'],
                history['ode_log_probs_per_step']
            )):
                steps_axis = list(range(len(div_curve)))
                ax_div.plot(steps_axis, div_curve, color=colors[idx], 
                           alpha=0.7, label=f'Ckpt {ckpt_step}', linewidth=1.5)
                ax_ode.plot(steps_axis, ode_curve, color=colors[idx], 
                           alpha=0.7, label=f'Ckpt {ckpt_step}', linewidth=1.5)
            
            ax_div.set_xlabel('Denoising Step')
            ax_div.set_ylabel('Divergence')
            ax_div.set_title(f'Divergence over Steps (Multi-checkpoint)\n{prompt_list[0][:50]}...')
            ax_div.legend(loc='best', fontsize=8)
            ax_div.grid(True, alpha=0.3)
            
            ax_ode.set_xlabel('Denoising Step')
            ax_ode.set_ylabel('ODE Log Prob')
            ax_ode.set_title(f'ODE Log Prob over Steps (Multi-checkpoint)\n{prompt_list[0][:50]}...')
            ax_ode.legend(loc='best', fontsize=8)
            ax_ode.grid(True, alpha=0.3)
            
            div_overlay_path = os.path.join(tmpdir, "divergence_overlay.png")
            ode_overlay_path = os.path.join(tmpdir, "ode_overlay.png")
            fig_div.tight_layout()
            fig_ode.tight_layout()
            fig_div.savefig(div_overlay_path, dpi=150)
            fig_ode.savefig(ode_overlay_path, dpi=150)
            plt.close(fig_div)
            plt.close(fig_ode)
        
        # === 方案B-2: 每个denoising step的值随checkpoint变化 ===
        if num_history > 1:
            div_matrix = np.array(history['divergences_per_step'])
            ode_matrix = np.array(history['ode_log_probs_per_step'])
            num_denoising_steps = div_matrix.shape[1]
            
            fig_div_b2, ax_div_b2 = plt.subplots(figsize=(12, 6))
            fig_ode_b2, ax_ode_b2 = plt.subplots(figsize=(12, 6))
            
            cmap = plt.cm.plasma
            colors = [cmap(i / (num_denoising_steps - 1)) for i in range(num_denoising_steps)]
            
            if num_denoising_steps > 20:
                step_indices_to_plot = [0, num_denoising_steps//4, num_denoising_steps//2, 
                                       3*num_denoising_steps//4, num_denoising_steps-1]
            elif num_denoising_steps > 10:
                step_indices_to_plot = list(range(0, num_denoising_steps, 2))
            else:
                step_indices_to_plot = list(range(num_denoising_steps))
            
            for step_idx in step_indices_to_plot:
                ax_div_b2.plot(history['checkpoint_steps'], div_matrix[:, step_idx], 
                              color=colors[step_idx], marker='o', markersize=3,
                              alpha=0.8, label=f'Step {step_idx}', linewidth=1.5)
                ax_ode_b2.plot(history['checkpoint_steps'], ode_matrix[:, step_idx], 
                              color=colors[step_idx], marker='o', markersize=3,
                              alpha=0.8, label=f'Step {step_idx}', linewidth=1.5)
            
            ax_div_b2.set_xlabel('Checkpoint Step')
            ax_div_b2.set_ylabel('Divergence')
            ax_div_b2.set_title(f'Divergence at Each Denoising Step across Checkpoints\n{prompt_list[0][:50]}...')
            ax_div_b2.legend(loc='best', fontsize=8, ncol=2)
            ax_div_b2.grid(True, alpha=0.3)
            
            ax_ode_b2.set_xlabel('Checkpoint Step')
            ax_ode_b2.set_ylabel('ODE Log Prob')
            ax_ode_b2.set_title(f'ODE Log Prob at Each Denoising Step across Checkpoints\n{prompt_list[0][:50]}...')
            ax_ode_b2.legend(loc='best', fontsize=8, ncol=2)
            ax_ode_b2.grid(True, alpha=0.3)
            
            div_b2_path = os.path.join(tmpdir, "divergence_over_checkpoints.png")
            ode_b2_path = os.path.join(tmpdir, "ode_over_checkpoints.png")
            fig_div_b2.tight_layout()
            fig_ode_b2.tight_layout()
            fig_div_b2.savefig(div_b2_path, dpi=150)
            fig_ode_b2.savefig(ode_b2_path, dpi=150)
            plt.close(fig_div_b2)
            plt.close(fig_ode_b2)
        
        # 记录到wandb（使用不同的前缀避免覆盖）
        log_dict = {
            f'checkpoint_viz_{history_key}/images': wandb_images,
            f'checkpoint_viz_{history_key}/divergence_curve': wandb.plot.line_series(
                xs=list(range(actual_num_steps)),
                ys=[divergences_np],
                keys=["Divergence"],
                title=f"Divergence over steps (ckpt {checkpoint_step})",
                xname="Step"
            ),
            f'checkpoint_viz_{history_key}/ode_log_prob_curve': wandb.plot.line_series(
                xs=list(range(actual_num_steps)),
                ys=[ode_log_probs_np],
                keys=["ODE Log Prob"],
                title=f"ODE Log Prob over steps (ckpt {checkpoint_step})",
                xname="Step"
            ),
        }
        
        if num_history > 1:
            log_dict[f'checkpoint_history_{history_key}/divergence_overlay'] = wandb.Image(div_overlay_path)
            log_dict[f'checkpoint_history_{history_key}/ode_overlay'] = wandb.Image(ode_overlay_path)
            log_dict[f'checkpoint_history_{history_key}/divergence_over_checkpoints'] = wandb.Image(div_b2_path)
            log_dict[f'checkpoint_history_{history_key}/ode_over_checkpoints'] = wandb.Image(ode_b2_path)
        
        wandb.log(log_dict, step=checkpoint_step)
    
    # 清理
    del images, all_latents, all_ode_log_probs, all_divergences
    torch.cuda.empty_cache()
    
    print(f"Visualization complete for checkpoint {checkpoint_step}")


def main(_):
    config = FLAGS.config
    
    # 初始化wandb
    wandb.init(
        project=FLAGS.wandb_project,
        name=FLAGS.wandb_run_name,
        config=config.to_dict(),
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 获取所有checkpoint路径
    checkpoint_pattern = os.path.join(FLAGS.checkpoint_dir, FLAGS.checkpoint_pattern)
    checkpoint_dirs = sorted(glob.glob(checkpoint_pattern))
    
    if not checkpoint_dirs:
        print(f"No checkpoints found matching pattern: {checkpoint_pattern}")
        return
    
    # 提取checkpoint步数
    checkpoint_info = []
    for ckpt_dir in checkpoint_dirs:
        match = re.search(r'checkpoint-(\d+)', os.path.basename(ckpt_dir))
        if match:
            step = int(match.group(1))
            checkpoint_info.append((step, ckpt_dir))
    
    checkpoint_info.sort(key=lambda x: x[0])
    
    # 过滤checkpoint
    if FLAGS.checkpoint_steps:
        selected_steps = [int(s) for s in FLAGS.checkpoint_steps]
        checkpoint_info = [(s, p) for s, p in checkpoint_info if s in selected_steps]
    elif not FLAGS.visualize_all:
        # 默认选择一些代表性的checkpoint（首、中、尾）
        if len(checkpoint_info) > 10:
            indices = [0, len(checkpoint_info)//4, len(checkpoint_info)//2, 
                      3*len(checkpoint_info)//4, len(checkpoint_info)-1]
            checkpoint_info = [checkpoint_info[i] for i in indices]
    
    print(f"Will visualize {len(checkpoint_info)} checkpoints:")
    for step, path in checkpoint_info:
        print(f"  - Step {step}: {path}")
    
    # 基础模型路径
    base_model_path = "/workspace/flow_grpo/models/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"
    
    # 准备测试prompts（与训练时保持一致）
    eval_dataset_200 = TextPromptDataset(config.eval_dataset, 'rand200')
    eval_dataset_20 = TextPromptDataset(config.eval_dataset, 'duplicate20')
    
    vis_prompts = [
        *eval_dataset_200.prompts[:4],  # 前4个来自dataset_200
        *eval_dataset_20.prompts[:4],   # 前4个来自dataset_20
    ]
    
    prompt_names = [
        "dataset200_prompt0", "dataset200_prompt1", "dataset200_prompt2", "dataset200_prompt3",
        "dataset20_prompt0", "dataset20_prompt1", "dataset20_prompt2", "dataset20_prompt3",
    ]
    
    # 缓存pipeline以避免重复加载基础模型
    cached_pipeline = None
    
    # 遍历每个checkpoint
    for ckpt_idx, (checkpoint_step, checkpoint_path) in enumerate(checkpoint_info):
        print(f"\n{'='*60}")
        print(f"Processing checkpoint {checkpoint_step} ({ckpt_idx + 1}/{len(checkpoint_info)})")
        print(f"{'='*60}")
        
        # 显示GPU内存使用情况
        if torch.cuda.is_available():
            print(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        # 加载模型（第一次加载基础模型，后续重用）
        if ckpt_idx == 0:
            # 第一次加载，创建pipeline
            pipeline = load_pipeline_with_checkpoint(checkpoint_path, base_model_path, device, pipeline_cache=None)
            cached_pipeline = pipeline  # 缓存基础模型
        else:
            # 后续checkpoint，卸载之前的LoRA并加载新的
            print("Unloading previous LoRA weights...")
            if hasattr(cached_pipeline.transformer, 'peft_config'):
                try:
                    # 卸载之前的LoRA adapter
                    cached_pipeline.transformer = cached_pipeline.transformer.unload()
                    torch.cuda.empty_cache()
                    print("Previous LoRA weights unloaded successfully")
                except Exception as e:
                    print(f"Warning: Failed to unload previous LoRA: {e}")
            
            # 加载新的LoRA权重
            lora_path = os.path.join(checkpoint_path, "lora")
            if os.path.exists(lora_path):
                print(f"Loading LoRA weights from {lora_path}")
                torch.cuda.empty_cache()
                cached_pipeline.transformer = PeftModel.from_pretrained(
                    cached_pipeline.transformer, 
                    lora_path,
                    is_trainable=False
                )
                cached_pipeline.transformer.set_adapter("default")
            
            pipeline = cached_pipeline
        
        if torch.cuda.is_available():
            print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
        tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
        
        # 可视化每个prompt
        for prompt, prompt_name in zip(vis_prompts, prompt_names):
            print(f"  Visualizing: {prompt_name}")
            visualize_denoising_from_checkpoint(
                pipeline, prompt, text_encoders, tokenizers, config, device,
                checkpoint_step=checkpoint_step,
                num_visualize_steps=10,
                prompt_name=prompt_name
            )
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 最后清理
    if cached_pipeline is not None:
        del cached_pipeline
        del pipeline
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print("="*60)
    wandb.finish()


if __name__ == "__main__":
    app.run(main)