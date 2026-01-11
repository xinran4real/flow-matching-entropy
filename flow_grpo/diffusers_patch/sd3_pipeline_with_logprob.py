# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
from typing import Any, Dict, List, Optional, Union
import torch
import math
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .sd3_sde_with_logprob import sde_step_with_logprob

@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_layer_guidance_scale: float = 2.8,
    noise_level: float = 0.7,
    decode_latents: bool = True,
    calculate_log_prob: bool = False,
    eval_log_prob: bool = False,
    visualize_steps: bool = False,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    ).float()

    ode_log_prob_final = None
    if eval_log_prob:
        # Assuming latents are sampled from standard Normal N(0, I)
        # latents shape: [Batch, Seq_Len, Dim] or [Batch, Channels, H, W] depending on packing
        # Flux usually works with packed latents [B, S, D]
        # log p(z) = -0.5 * ( ||z||^2 + d * log(2pi) )
        ode_log_prob_final = -0.5 * (
            latents**2 + math.log(2 * math.pi)
        )
        ode_log_prob_final = ode_log_prob_final.mean(dim=tuple(range(1, ode_log_prob_final.ndim)))
        # Set calculate_log_prob to True, in case it is not provided
        calculate_log_prob = True

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []
    all_epsilon = []
    all_ode_log_probs = []
    if visualize_steps:
        all_divergences = []
        # Set calculate_log_prob and eval_log_prob to True, in case it is not provided
        calculate_log_prob = True
        eval_log_prob = True

    # 7. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            # noise_pred = noise_pred.to(prompt_embeds.dtype)
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents_dtype = latents.dtype

            # --- [MODIFIED] Calculate Divergence and Update Log Prob ---
            if calculate_log_prob:
                # Hutchinson Trace Estimator with Finite Difference
                # div(v) = Tr(J) = E[epsilon^T J epsilon]
                # J epsilon approx (v(x + h*epsilon) - v(x)) / h
                
                # A. Sample Rademacher random vector epsilon
                epsilon = (torch.randn_like(latents).float().to(latents_dtype))
                # epsilon = torch.randint(0, 2, latents.shape, device=latents.device, dtype=latents_dtype) * 2 - 1
                
                # B. Finite Difference Step size
                # Small enough for approximation, large enough to avoid numerical noise
                h = 1e-3 
                
                # C. Perturbed input
                if self.do_classifier_free_guidance:
                    latents_perturbed = latent_model_input + h * torch.cat([epsilon] * 2)
                else:
                    latents_perturbed = latent_model_input + h * epsilon
                
                # D. Calculate v(x + h*epsilon)
                noise_pred_perturbed = self.transformer(
                    hidden_states=latents_perturbed,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                if self.do_classifier_free_guidance:
                    noise_pred_perturbed_uncond, noise_pred_perturbed_text = noise_pred_perturbed.chunk(2)
                    noise_pred_perturbed = (
                        noise_pred_perturbed_uncond
                        + self.guidance_scale
                        * (noise_pred_perturbed_text - noise_pred_perturbed_uncond)
                    )

                # E. Approximate JVP: J * epsilon
                # Using forward difference: (f(x+h) - f(x)) / h
                jvp = (noise_pred_perturbed - noise_pred) / h
                
                # F. Estimate Trace: epsilon^T * jvp
                # Sum over all dimensions except batch
                # flatten_dims = tuple(range(1, latents.ndim))
                divergence_est = epsilon * jvp # torch.sum(, dim=flatten_dims)
                
                # G. The ODE for log_prob is: d(log_p)/dt = -div(v)
                # So the "velocity" for log_prob is -divergence_est
                log_prob_velocity = -divergence_est

            if calculate_log_prob:
                latents, log_prob, prev_latents_mean, std_dev_t, ode_log_prob = sde_step_with_logprob(
                    self.scheduler, 
                    noise_pred.float(), 
                    t.unsqueeze(0), 
                    latents.float(),
                    noise_level=noise_level,
                    log_prob_velocity=log_prob_velocity,
                )
            else:
                latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                    self.scheduler, 
                    noise_pred.float(), 
                    t.unsqueeze(0), 
                    latents.float(),
                    noise_level=noise_level,
                )
            
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            if calculate_log_prob:
                all_epsilon.append(epsilon)
                ode_log_prob = ode_log_prob.mean(dim=tuple(range(1, ode_log_prob.ndim)))
                all_ode_log_probs.append(ode_log_prob)
            if eval_log_prob:
                # log_p_{t-1} = log_p_t + dt * (-div_t)
                ode_log_prob_final = ode_log_prob_final + ode_log_prob
            if visualize_steps:
                # 计算divergence的标量, 对空间维度求和，保留 batch 维度
                div_scalar = divergence_est.sum(dim=tuple(range(1, divergence_est.ndim)))
                all_divergences.append(div_scalar)
            # if latents.dtype != latents_dtype:
            #     latents = latents.to(latents_dtype)
            
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
    if decode_latents:
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        latents = latents.to(dtype=self.vae.dtype)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
    else:
        image = latents
    # Offload all models
    self.maybe_free_model_hooks()

    if visualize_steps:
        return image, all_latents, all_log_probs, all_epsilon, all_ode_log_probs, all_divergences, ode_log_prob_final
    elif eval_log_prob:
        return image, all_latents, all_log_probs, all_epsilon, all_ode_log_probs, ode_log_prob_final
    elif calculate_log_prob:
        return image, all_latents, all_log_probs, all_epsilon, all_ode_log_probs
    return image, all_latents, all_log_probs