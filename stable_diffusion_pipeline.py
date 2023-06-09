#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import inspect
from typing import List, Tuple, Union, Optional, Callable, Dict

import diffusers
from diffusers.configuration_utils import FrozenDict
from tqdm import tqdm
from cuda import cudart
from PIL import Image
import gc
from lpw import LongPromptWeightingPipeline
from models import make_CLIP, make_tokenizer, make_UNet, make_VAE, make_VAEEncoder
import numpy as np
import os
import onnx
from polygraphy import cuda
import torch

from text_encoder import TensorRTCLIPTextModel
from utilities import Engine, device_view, save_image, preprocess_image
from utilities import DPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler

class StableDiffusionPipeline:
    """
    Application showcasing the acceleration of Stable Diffusion Txt2Img v1.4, v1.5, v2.0-base, v2.0, v2.1, v2.1-base pipeline using NVidia TensorRT w/ Plugins.
    """
    def __init__(
        self,
        version="1.5",
        inpaint=False,
        stages=['clip', 'unet', 'vae'],
        max_batch_size=4,
        denoising_steps=50,
        scheduler="DPM",
        guidance_scale=7.5,
        device='cuda',
        output_dir='.',
        hf_token=None,
        verbose=False,
        nvtx_profile=False,
    ):
        """
        Initializes the Diffusion pipeline.

        Args:
            version (str):
                The version of the pipeline. Should be one of [1.4, 1.5, 2.0, 2.0-base, 2.1, 2.1-base]
            inpaint (bool):
                True if inpainting pipeline.
            stages (list):
                Ordered sequence of stages. Options: ['vae_encoder', 'clip','unet','vae']
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            denoising_steps (int):
                The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the expense of slower inference.
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of [DDIM, DPM, EulerA, LMSD, PNDM].
            guidance_scale (float):
                Guidance scale is enabled by setting as > 1.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
            device (str):
                PyTorch device to run inference. Default: 'cuda'
            output_dir (str):
                Output directory for log files and image artifacts
            hf_token (str):
                HuggingFace User Access Token to use for downloading Stable Diffusion model checkpoints.
            verbose (bool):
                Enable verbose logging.
            nvtx_profile (bool):
                Insert NVTX profiling markers.
        """

        self.output_dir = output_dir
        self.hf_token = hf_token
        self.device = device
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile
        self.max_batch_size = max_batch_size
        self.version = version

        self.stages = stages
        self.inpaint = inpaint

        self.stream = None # loaded in loadResources()
        self.tokenizer = None # loaded in loadResources()
        self.models = {} # loaded in loadEngines()
        self.engine = {} # loaded in loadEngines()

    def loadResources(self, image_height, image_width, batch_size, seed, denoising_step, guidance_scal, scheduler):

        self.denoising_steps = denoising_step
        assert guidance_scal > 1.0
        self.guidance_scale = guidance_scal
        sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}
        SCHEDULERS = {
            "ddim": diffusers.DDIMScheduler,
            "deis": diffusers.DEISMultistepScheduler,
            "dpm2": diffusers.KDPM2DiscreteScheduler,
            "dpm2-a": diffusers.KDPM2AncestralDiscreteScheduler,
            "euler_a": diffusers.EulerAncestralDiscreteScheduler,
            "euler": diffusers.EulerDiscreteScheduler,
            "heun": diffusers.DPMSolverMultistepScheduler,
            "dpm++": diffusers.DPMSolverMultistepScheduler,
            "dpm": diffusers.DPMSolverMultistepScheduler,
            "pndm": diffusers.PNDMScheduler,
        }

        conf = FrozenDict([('num_train_timesteps', 1000), ('beta_start', 0.00085), ('beta_end', 0.012)])

        self.scheduler = SCHEDULERS[scheduler].from_config(conf)

        # sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}
        #
        # if scheduler == "DDIM":
        #     self.scheduler = DDIMScheduler(device=self.device, **sched_opts)
        # elif scheduler == "DPM":
        #     self.scheduler = DPMScheduler(device=self.device, **sched_opts)
        # elif scheduler == "EulerA":
        #     self.scheduler = EulerAncestralDiscreteScheduler(device=self.device, **sched_opts)
        # elif scheduler == "LMSD":
        #     self.scheduler = LMSDiscreteScheduler(device=self.device, **sched_opts)
        # elif scheduler == "PNDM":
        #     sched_opts["steps_offset"] = 1
        #     self.scheduler = PNDMScheduler(device=self.device, **sched_opts)
        # else:
        #     raise ValueError(f"Scheduler should be either DDIM, DPM, EulerA, LMSD or PNDM")

        # Initialize noise generator
        self.generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

        # Pre-compute latent input scales and linear multistep coefficients
        self.scheduler.set_timesteps(self.denoising_steps, device=torch.device("cuda"))

        # self.scheduler.configure()

        # Create CUDA events and stream
        self.events = {}
        for stage in ['clip', 'denoise', 'vae', 'vae_encoder']:
            for marker in ['start', 'stop']:
                self.events[stage+'-'+marker] = cudart.cudaEventCreate()[1]
        self.stream = cuda.Stream()

        for model_name, obj in self.models.items():
            self.engine[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(batch_size, image_height, image_width),
                device=torch.device("cuda"),
            )

        self.text_encoder = TensorRTCLIPTextModel(
            self.engine["clip"], self.stream
        )
        self.lpw = LongPromptWeightingPipeline(
            self.text_encoder, self.tokenizer, self.device
        )

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e)

        for engine in self.engine.values():
            del engine

        self.stream.free()
        del self.stream

    def cachedModelName(self, model_name):
        if self.inpaint:
            model_name += '_inpaint'
        return model_name

    def getOnnxPath(self, model_name, onnx_dir, opt=True):
        return os.path.join(onnx_dir, self.cachedModelName(model_name)+('.opt' if opt else '')+'.onnx')

    def getEnginePath(self, model_name, engine_dir):
        return os.path.join(engine_dir, self.cachedModelName(model_name)+'.plan')

    def loadEngines(
        self,
        engine_dir
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.

        Args:
            engine_dir (str):
                Directory to write the TensorRT engines.
            onnx_dir (str):
                Directory to write the ONNX models.
            onnx_opset (int):
                ONNX opset version to export the models.
            opt_batch_size (int):
                Batch size to optimize for during engine building.
            opt_image_height (int):
                Image height to optimize for during engine building. Must be a multiple of 8.
            opt_image_width (int):
                Image width to optimize for during engine building. Must be a multiple of 8.
            force_export (bool):
                Force re-exporting the ONNX models.
            force_optimize (bool):
                Force re-optimizing the ONNX models.
            force_build (bool):
                Force re-building the TensorRT engine.
            static_batch (bool):
                Build engine only for specified opt_batch_size.
            static_shape (bool):
                Build engine only for specified opt_image_height & opt_image_width. Default = True.
            enable_refit (bool):
                Build engines with refit option enabled.
            enable_preview (bool):
                Enable TensorRT preview features.
            enable_all_tactics (bool):
                Enable all tactic sources during TensorRT engine builds.
            timing_cache (str):
                Path to the timing cache to accelerate build or None
            onnx_refit_dir (str):
                Directory containing refit ONNX models.
        """
        # Load text tokenizer
        self.tokenizer = make_tokenizer(self.version, self.hf_token)

        # Load pipeline models
        models_args = {'version': self.version, 'hf_token': self.hf_token, 'device': self.device, 'verbose': self.verbose, 'max_batch_size': self.max_batch_size}
        if 'vae_encoder' in self.stages:
            self.models['vae_encoder'] = make_VAEEncoder(inpaint=self.inpaint, **models_args)
        if 'clip' in self.stages:
            self.models['clip'] = make_CLIP(inpaint=self.inpaint, **models_args)
        if 'unet' in self.stages:
            self.models['unet'] = make_UNet(inpaint=self.inpaint, **models_args)
        if 'vae' in self.stages:
            self.models['vae'] = make_VAE(inpaint=self.inpaint, **models_args)

        # Build TensorRT engines
        for model_name, obj in self.models.items():
            engine_path = self.getEnginePath(model_name, engine_dir)
            engine = Engine(engine_path)
            self.engine[model_name] = engine

        # Load and activate TensorRT engines
        for model_name, obj in self.models.items():
            engine = self.engine[model_name]
            engine.load()
            engine.activate()

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream)


    def initialize_latents(
        self,
        batch_size: int,
        unet_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Union[torch.Generator, List[torch.Generator]],
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (batch_size, unet_channels, height, width)
        latents = diffusers.utils.randn_tensor(
            shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def initialize_timesteps(
        self, num_inference_steps: int, strength: int, device: torch.device
    ):
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        offset = (
            self.scheduler.steps_offset
            if hasattr(self.scheduler, "steps_offset")
            else 0
        )
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        num_inference_steps = self.scheduler.timesteps[t_start:].to(device)
        return num_inference_steps, t_start

    def prepare_extra_step_kwargs(self, generator: torch.Generator, eta: float):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def preprocess_images(
        self, batch_size: int, device: torch.device, images: Tuple[Image.Image] = ()
    ):
        init_images: List[torch.Tensor] = []
        for image in images:
            if isinstance(image, torch.Tensor):
                init_images.append(image)
                continue
            image = preprocess_image(image)
            image = image.to(device).float()
            image = image.repeat(batch_size, 1, 1, 1)
            init_images.append(image)
        return tuple(init_images)



    def encode_prompt(self, prompt, negative_prompt):
        cudart.cudaEventRecord(self.events['clip-start'], 0)

        text_embeddings = self.lpw(
                prompt,
                negative_prompt,
                1,
                max_embeddings_multiples=1,
            ).to(dtype=torch.float16)

        cudart.cudaEventRecord(self.events['clip-stop'], 0)

        return text_embeddings

    def run_engine(self, model_name: str, feed_dict: Dict[str, cuda.DeviceView]):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream)

    def denoise_latent(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        timesteps=None,
        step_offset=0,
        guidance_scale: int = 7.5,
        mask: Optional[torch.Tensor] = None,
        masked_image_latents: Optional[torch.Tensor] = None,
        extra_step_kwargs: dict = {},
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        cudart.cudaEventRecord(self.events["denoise-start"], 0)
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps
        for step, timestep in enumerate(tqdm(timesteps)):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # Predict the noise residual
            timestep_float = (
                timestep.float() if timestep.dtype != torch.float32 else timestep
            )

            sample_inp = device_view(latent_model_input)
            timestep_inp = device_view(timestep_float)
            embeddings_inp = device_view(text_embeddings)
            noise_pred = self.run_engine(
                "unet",
                {
                    "sample": sample_inp,
                    "timestep": timestep_inp,
                    "encoder_hidden_states": embeddings_inp,
                },
            )["latent"]

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latents = self.scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=latents,
                **extra_step_kwargs,
            ).prev_sample

        latents = 1.0 / 0.18215 * latents
        cudart.cudaEventRecord(self.events["denoise-stop"], 0)
        return latents

    # def denoise_latent(self, latents, text_embeddings, timesteps=None, step_offset=0, mask=None, masked_image_latents=None):
    #     cudart.cudaEventRecord(self.events['denoise-start'], 0)
    #     if not isinstance(timesteps, torch.Tensor):
    #         timesteps = self.scheduler.timesteps
    #     for step_index, timestep in enumerate(timesteps):
    #
    #         # Expand the latents if we are doing classifier free guidance
    #         latent_model_input = torch.cat([latents] * 2)
    #         latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_offset + step_index, timestep)
    #         if isinstance(mask, torch.Tensor):
    #             latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
    #
    #         # Predict the noise residual
    #
    #         embeddings_dtype = np.float16
    #         timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep
    #
    #         sample_inp = device_view(latent_model_input)
    #         timestep_inp = device_view(timestep_float)
    #         embeddings_inp = device_view(text_embeddings)
    #         noise_pred = self.runEngine('unet', {"sample": sample_inp, "timestep": timestep_inp, "encoder_hidden_states": embeddings_inp})['latent']
    #
    #         # Perform guidance
    #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #         noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
    #
    #         latents = self.scheduler.step(noise_pred, latents, step_offset + step_index, timestep)
    #
    #
    #     latents = 1. / 0.18215 * latents
    #     cudart.cudaEventRecord(self.events['denoise-stop'], 0)
    #     return latents

    def encode_image(self, init_image):
        cudart.cudaEventRecord(self.events['vae_encoder-start'], 0)
        init_latents = self.runEngine('vae_encoder', {"images": device_view(init_image)})['latent']
        cudart.cudaEventRecord(self.events['vae_encoder-stop'], 0)

        init_latents = 0.18215 * init_latents
        return init_latents

    def decode_latent(self, latents):
        cudart.cudaEventRecord(self.events['vae-start'], 0)
        images = self.runEngine('vae', {"latent": device_view(latents)})['images']
        cudart.cudaEventRecord(self.events['vae-stop'], 0)
        return images

    def print_summary(self, denoising_steps, tic, toc, vae_enc=False):
            print('|------------|--------------|')
            print('| {:^10} | {:^12} |'.format('Module', 'Latency'))
            print('|------------|--------------|')
            if vae_enc:
                print('| {:^10} | {:>9.2f} ms |'.format('VAE-Enc', cudart.cudaEventElapsedTime(self.events['vae_encoder-start'], self.events['vae_encoder-stop'])[1]))
            print('| {:^10} | {:>9.2f} ms |'.format('CLIP', cudart.cudaEventElapsedTime(self.events['clip-start'], self.events['clip-stop'])[1]))
            print('| {:^10} | {:>9.2f} ms |'.format('UNet x '+str(denoising_steps), cudart.cudaEventElapsedTime(self.events['denoise-start'], self.events['denoise-stop'])[1]))
            print('| {:^10} | {:>9.2f} ms |'.format('VAE-Dec', cudart.cudaEventElapsedTime(self.events['vae-start'], self.events['vae-stop'])[1]))
            print('|------------|--------------|')
            print('| {:^10} | {:>9.2f} ms |'.format('Pipeline', (toc - tic)*1000.))
            print('|------------|--------------|')

    def save_image(self, images, pipeline, prompt):
            # Save image
            image_name_prefix = pipeline+'-fp16'+''.join(set(['-'+prompt[i].replace(' ','_')[:10] for i in range(len(prompt))]))+'-'
            save_image(images, self.output_dir, image_name_prefix)
