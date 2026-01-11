from collections import defaultdict
import os
import time
import tempfile
import hashlib
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
from PIL import Image
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

os.environ['WANDB_API_KEY'] = '7de18aa30d0f35639b8fcd34853791049d12b109'

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

# 辅助函数：可视化latents
def visualize_latents(latents, vae):
    """
    将latents转换为可视化的图像
    Args:
        latents: torch.Tensor, shape [batch, channels, h, w]
        vae: VAE model
    Returns:
        images: torch.Tensor, shape [batch, 3, h, w]
    """
    with torch.no_grad():
        latents_scaled = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        latents_scaled = latents_scaled.to(dtype=vae.dtype)
        images = vae.decode(latents_scaled, return_dict=False)[0]
        # images should be in range [-1, 1] or [0, 1], convert to [0, 1]
        images = (images + 1) / 2.0 if images.min() < 0 else images
        images = images.clamp(0, 1)
    return images

def eval_ode_log_prob(pipeline, eval_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_ode_log_prob_finals = []
    all_rewards = defaultdict(list)
    all_images = []
    all_prompts = []
    for test_batch in tqdm(
            eval_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
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
                    eval_log_prob=config.eval_ode_log_prob,
                    visualize_steps=config.visualize_steps,
                )
        for m in prompt_metadata:
            m['requires_grad'] = False

        ode_log_prob_finals = accelerator.gather(ode_log_prob_final)
        all_ode_log_prob_finals.append(ode_log_prob_finals.cpu().numpy())

        # Compute rewards
        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False, use_reward_gradient=config.use_reward_gradient)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)

        if config.visualize_steps:
            images_gathered = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu()
            all_images.append(images_gathered)
            
            # Gather prompts
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            prompt_ids_gathered = accelerator.gather(prompt_ids).cpu().numpy()
            prompts_gathered = pipeline.tokenizer.batch_decode(prompt_ids_gathered, skip_special_tokens=True)
            all_prompts.extend(prompts_gathered)            

    if accelerator.is_main_process:
        # 记录最终的 ODE_LOG_PROB 统计信息
        # 转换为numpy数组以便统计
        ode_log_prob_array = np.concatenate(all_ode_log_prob_finals, axis=0)

        # 统计基本信息
        total_samples = len(ode_log_prob_array)
        
        # 统计大于0.9的样本
        high_prob_mask = ode_log_prob_array > 0.9
        high_prob_count = np.sum(high_prob_mask)
        high_prob_ratio = high_prob_count / total_samples if total_samples > 0 else 0
        
        # 统计小于0.3的样本
        low_prob_mask = ode_log_prob_array < 0.3
        low_prob_count = np.sum(low_prob_mask)
        low_prob_ratio = low_prob_count / total_samples if total_samples > 0 else 0

        # 计算均值、标准差、最小值、最大值
        ode_log_prob_mean = np.mean(ode_log_prob_array)
        ode_log_prob_std = np.std(ode_log_prob_array)
        ode_log_prob_min = np.min(ode_log_prob_array)
        ode_log_prob_max = np.max(ode_log_prob_array)
        ode_log_prob_median = np.median(ode_log_prob_array)
        
        # 计算变异系数
        coeff_of_variation = ode_log_prob_std / ode_log_prob_mean if ode_log_prob_mean != 0 else float('inf')
        
        # 打印详细统计信息
        print("\n" + "="*60)
        print("ODE_LOG_PROB 详细统计信息")
        print("="*60)
        print(f"样本总数: {total_samples}")
        print(f"均值: {ode_log_prob_mean:.4f}")
        print(f"标准差: {ode_log_prob_std:.4f}")
        print(f"变异系数(CV): {coeff_of_variation:.3f} ({coeff_of_variation*100:.1f}%)")
        print(f"最小值: {ode_log_prob_min:.4f}")
        print(f"最大值: {ode_log_prob_max:.4f}")
        print(f"中位数: {ode_log_prob_median:.4f}")
        print("-"*60)
        print(f"大于0.9的高质量样本: {high_prob_count}/{total_samples} ({high_prob_ratio*100:.1f}%)")
        print(f"小于0.3的低质量样本: {low_prob_count}/{total_samples} ({low_prob_ratio*100:.1f}%)")
        print("="*60)
        # 将统计信息记录到wandb
        wandb.log({
            'ode_log_prob_finals': ode_log_prob_mean,
            'ode_log_prob_std': ode_log_prob_std,
            'ode_log_prob_min': ode_log_prob_min,
            'ode_log_prob_max': ode_log_prob_max,
            'ode_log_prob_median': ode_log_prob_median,
            'ode_log_prob_cv': coeff_of_variation,
            'ode_log_prob_high_ratio': high_prob_ratio,
            'ode_log_prob_low_ratio': low_prob_ratio,
        }, step=global_step)

        if config.visualize_steps and len(all_images) > 0:
            # 记录最终图像
            log_dict = {}
            all_rewards_concat = {key: np.concatenate(value) for key, value in all_rewards.items()}
            all_images_concat = torch.cat(all_images, dim=0).numpy()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(10, len(all_images_concat))  # 最多记录10张
                
                for idx in range(num_samples):
                    image = all_images_concat[idx]
                    pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                
                # 准备caption
                wandb_images = []
                for idx in range(num_samples):
                    prompt = all_prompts[idx]
                    ode_prob = ode_log_prob_array[idx]
                    reward_str = " | ".join(
                        f"{k}: {all_rewards_concat[k][idx]:.2f}" 
                        for k in all_rewards_concat 
                        if all_rewards_concat[k][idx] != -10
                    )
                    caption = f"{prompt:.200}\nODE_log_prob: {ode_prob:.4f} | {reward_str}"
                    
                    wandb_images.append(
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=caption
                        )
                    )
                
                log_dict[f'ode_images'] = wandb_images
                wandb.log(log_dict, step=global_step)
            
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

# 全局字典，用于存储历史数据（键为prompt的hash）
_visualization_history = defaultdict(lambda: {
    'global_steps': [],
    'ode_log_probs_per_step': [],  # List of arrays, each array is ode_log_prob at each denoising step
    'divergences_per_step': []  # List of arrays, each array is divergence at each denoising step
})

# 新增函数：可视化中间步骤
def visualize_denoising_process(pipeline, prompts, text_encoders, tokenizers, config, accelerator, 
                                global_step, num_visualize_steps=10, prompt_name=""):
    """
    可视化去噪过程中的中间步骤
    
    Args:
        prompts: str, 要可视化的prompt
        num_visualize_steps: int, 要可视化的步骤数量
        prompt_name: str, prompt的标识名称（用于区分不同prompt）
    """

    pipeline.transformer.eval()
    
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    
    # 处理单个prompt
    prompt = [prompts] if isinstance(prompts, str) else [prompts[0]]
    
    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
        prompt, 
        text_encoders, 
        tokenizers, 
        max_sequence_length=128, 
        device=accelerator.device
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
    
    # 提取数据
    divergences_np = torch.stack([d[0] for d in all_divergences]).cpu().numpy()
    ode_log_probs_np = torch.stack([o[0] for o in all_ode_log_probs]).cpu().numpy()
    
    actual_num_steps = len(divergences_np)
    
    # 存储历史数据
    prompt_hash = hashlib.md5(prompt[0].encode()).hexdigest()[:8]
    if prompt_name:
        history_key = f"{prompt_name}_{prompt_hash}"
    else:
        history_key = prompt_hash
    
    _visualization_history[history_key]['global_steps'].append(global_step)
    _visualization_history[history_key]['ode_log_probs_per_step'].append(ode_log_probs_np)
    _visualization_history[history_key]['divergences_per_step'].append(divergences_np)
    
    # 选择要可视化的步骤（等距采样）
    total_steps = len(all_latents) - 1  # 减1因为第一个是初始噪声
    if total_steps <= num_visualize_steps:
        step_indices = list(range(total_steps))
    else:
        step_indices = [int(i * total_steps / (num_visualize_steps - 1)) for i in range(num_visualize_steps - 1)]
        step_indices.append(total_steps - 1)  # 确保包含最后一步
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wandb_images = []
        
        for vis_idx, step_idx in enumerate(step_indices):
            # 可视化latent
            latent = all_latents[step_idx]
            vis_image = visualize_latents(latent, pipeline.vae)
            
            # 转换为numpy并保存
            vis_image_np = (vis_image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pil = Image.fromarray(vis_image_np)
            pil = pil.resize((config.resolution, config.resolution))
            pil.save(os.path.join(tmpdir, f"step_{vis_idx}.jpg"))
            
            # 获取对应的统计信息
            if step_idx > 0:  # 第0步是初始噪声，没有对应的统计
                ode_log_prob = all_ode_log_probs[step_idx - 1][0].item()
                divergence = all_divergences[step_idx - 1][0].item()
                caption = f"Step {step_idx}/{total_steps}\nODE_log_prob: {ode_log_prob:.4f}\nDivergence: {divergence:.4f}"
            else:
                caption = f"Step {step_idx}/{total_steps} (Initial Noise)"
            
            wandb_images.append(
                wandb.Image(
                    os.path.join(tmpdir, f"step_{vis_idx}.jpg"),
                    caption=caption
                )
            )
        
        # 记录最终图像
        final_image_np = (images[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil = Image.fromarray(final_image_np)
        pil = pil.resize((config.resolution, config.resolution))
        pil.save(os.path.join(tmpdir, "final.jpg"))
        
        wandb_images.append(
            wandb.Image(
                os.path.join(tmpdir, "final.jpg"),
                caption=f"Final Image\nPrompt: {prompt[0][:100]}"
            )
        )
        
        # ======= 叠加历史曲线（所有global_steps的曲线在一张图上）=======
        history = _visualization_history[history_key]
        num_history = len(history['global_steps'])
        
        if num_history > 1:
            # 生成渐变色（从浅到深）
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            fig_div, ax_div = plt.subplots(figsize=(10, 6))
            fig_ode, ax_ode = plt.subplots(figsize=(10, 6))
            
            # 创建colormap
            cmap = plt.cm.viridis
            colors = [cmap(i / (num_history - 1)) for i in range(num_history)]
            
            for idx, (step, div_curve, ode_curve) in enumerate(zip(
                history['global_steps'],
                history['divergences_per_step'],
                history['ode_log_probs_per_step']
            )):
                steps_axis = list(range(len(div_curve)))
                
                ax_div.plot(steps_axis, div_curve, color=colors[idx], 
                           alpha=0.7, label=f'Step {step}', linewidth=1.5)
                ax_ode.plot(steps_axis, ode_curve, color=colors[idx], 
                           alpha=0.7, label=f'Step {step}', linewidth=1.5)
            
            ax_div.set_xlabel('Denoising Step')
            ax_div.set_ylabel('Divergence')
            ax_div.set_title(f'Divergence over Steps (Multi-epoch)\n{prompt[0][:50]}...')
            ax_div.legend(loc='best', fontsize=8)
            ax_div.grid(True, alpha=0.3)
            
            ax_ode.set_xlabel('Denoising Step')
            ax_ode.set_ylabel('ODE Log Prob')
            ax_ode.set_title(f'ODE Log Prob over Steps (Multi-epoch)\n{prompt[0][:50]}...')
            ax_ode.legend(loc='best', fontsize=8)
            ax_ode.grid(True, alpha=0.3)
            
            # 保存到临时文件
            div_overlay_path = os.path.join(tmpdir, "divergence_overlay.png")
            ode_overlay_path = os.path.join(tmpdir, "ode_overlay.png")
            fig_div.tight_layout()
            fig_ode.tight_layout()
            fig_div.savefig(div_overlay_path, dpi=150)
            fig_ode.savefig(ode_overlay_path, dpi=150)
            plt.close(fig_div)
            plt.close(fig_ode)
        
        # ======= 每个denoising step的值随training epoch变化 =======
        if num_history > 1:
            import matplotlib.pyplot as plt
            
            # 将数据转换为 [num_epochs, num_denoising_steps]
            div_matrix = np.array(history['divergences_per_step'])  # shape: [num_epochs, num_steps]
            ode_matrix = np.array(history['ode_log_probs_per_step'])
            
            num_denoising_steps = div_matrix.shape[1]
            
            fig_div_b2, ax_div_b2 = plt.subplots(figsize=(12, 6))
            fig_ode_b2, ax_ode_b2 = plt.subplots(figsize=(12, 6))
            
            # 创建colormap (按denoising step)
            cmap = plt.cm.plasma
            colors = [cmap(i / (num_denoising_steps - 1)) for i in range(num_denoising_steps)]
            
            # 选择一些代表性的steps来绘制（避免太密集）
            if num_denoising_steps > 20:
                step_indices_to_plot = [0, num_denoising_steps//4, num_denoising_steps//2, 
                                       3*num_denoising_steps//4, num_denoising_steps-1]
            elif num_denoising_steps > 10:
                step_indices_to_plot = list(range(0, num_denoising_steps, 2))
            else:
                step_indices_to_plot = list(range(num_denoising_steps))
            
            for step_idx in step_indices_to_plot:
                ax_div_b2.plot(history['global_steps'], div_matrix[:, step_idx], 
                              color=colors[step_idx], marker='o', markersize=3,
                              alpha=0.8, label=f'Step {step_idx}', linewidth=1.5)
                ax_ode_b2.plot(history['global_steps'], ode_matrix[:, step_idx], 
                              color=colors[step_idx], marker='o', markersize=3,
                              alpha=0.8, label=f'Step {step_idx}', linewidth=1.5)
            
            ax_div_b2.set_xlabel('Global Step')
            ax_div_b2.set_ylabel('Divergence')
            ax_div_b2.set_title(f'Divergence at Each Denoising Step over Training\n{prompt[0][:50]}...')
            ax_div_b2.legend(loc='best', fontsize=8, ncol=2)
            ax_div_b2.grid(True, alpha=0.3)
            
            ax_ode_b2.set_xlabel('Global Step')
            ax_ode_b2.set_ylabel('ODE Log Prob')
            ax_ode_b2.set_title(f'ODE Log Prob at Each Denoising Step over Training\n{prompt[0][:50]}...')
            ax_ode_b2.legend(loc='best', fontsize=8, ncol=2)
            ax_ode_b2.grid(True, alpha=0.3)
            
            # 保存
            div_b2_path = os.path.join(tmpdir, "divergence_over_training.png")
            ode_b2_path = os.path.join(tmpdir, "ode_over_training.png")
            fig_div_b2.tight_layout()
            fig_ode_b2.tight_layout()
            fig_div_b2.savefig(div_b2_path, dpi=150)
            fig_ode_b2.savefig(ode_b2_path, dpi=150)
            plt.close(fig_div_b2)
            plt.close(fig_ode_b2)
        
        # 记录到wandb
        log_dict = {
            f'denoising_process_{history_key}/images': wandb_images,
            f'denoising_process_{history_key}/divergence_curve': wandb.plot.line_series(
                xs=list(range(1, actual_num_steps)), 
                ys=[divergences_np],
                keys=["Divergence"],
                title=f"Divergence over steps (epoch {global_step})",
                xname="Step"
            ),
            f'denoising_process_{history_key}/ode_log_prob_curve': wandb.plot.line_series(
                xs=list(range(1, actual_num_steps)),
                ys=[ode_log_probs_np],
                keys=["ODE Log Prob"],
                title=f"ODE Log Prob over steps (epoch {global_step})",
                xname="Step"
            ),
        }
        
        # 添加叠加图和历史图
        if num_history > 1:
            log_dict[f'denoising_history_{history_key}/divergence_overlay'] = wandb.Image(div_overlay_path)
            log_dict[f'denoising_history_{history_key}/ode_overlay'] = wandb.Image(ode_overlay_path)
            log_dict[f'denoising_history_{history_key}/divergence_over_training'] = wandb.Image(div_b2_path)
            log_dict[f'denoising_history_{history_key}/ode_over_training'] = wandb.Image(ode_b2_path)
        
        wandb.log(log_dict, step=global_step)
