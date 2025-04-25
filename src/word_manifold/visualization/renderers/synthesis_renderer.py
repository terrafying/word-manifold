"""
Image Synthesis Renderer for ASCII Patterns

Combines ASCII art with image synthesis techniques to create high-quality animations.
Features:
- ASCII pattern to image conversion
- Image-to-image synthesis
- Recursive upsampling
- Style transfer
- Inpainting for detail enhancement
- Audio-reactive modifications
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
import torchvision.transforms as T
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionUpscalePipeline
)
from typing import Optional, List, Dict, Tuple, Union
import logging
from pathlib import Path
import colorsys
import cv2

logger = logging.getLogger(__name__)

class SynthesisRenderer:
    """Renders ASCII patterns as high-quality synthesized images."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_id: str = "stabilityai/stable-diffusion-2-1",
        upscaler_id: str = "stabilityai/stable-diffusion-x4-upscaler",
        batch_size: int = 1,
        output_size: Tuple[int, int] = (1024, 1024),
        fps: int = 30,
        style_prompt: str = "detailed mystical mandala, intricate patterns, sacred geometry",
        negative_prompt: str = "text, watermark, signature, blurry, low quality",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        strength: float = 0.7
    ):
        """Initialize the synthesis renderer."""
        self.device = device
        self.output_size = output_size
        self.fps = fps
        self.style_prompt = style_prompt
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.batch_size = batch_size
        
        # Initialize models
        logger.info("Loading image synthesis models...")
        self.img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        self.upscaler = StableDiffusionUpscalePipeline.from_pretrained(
            upscaler_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        # Image processing
        self.transform = T.Compose([
            T.Resize(output_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        
        # Frame interpolation model for smooth transitions
        self.frame_interpolator = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
        )
        
        # Cache for frame transitions
        self.last_frame = None
        self.transition_frames = []
        
    def ascii_to_image(
        self,
        pattern: str,
        color: Optional[str] = None,
        background: Optional[str] = None
    ) -> Image.Image:
        """Convert ASCII pattern to initial image."""
        # Create base image
        width, height = self.output_size
        img = Image.new('RGB', (width, height), background or '#000000')
        draw = ImageDraw.Draw(img)
        
        # Calculate character size to fill image
        lines = pattern.split('\n')
        char_width = width // len(lines[0])
        char_height = height // len(lines)
        
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", char_height)
        except:
            font = ImageFont.load_default()
        
        # Draw pattern
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char != ' ':
                    pos = (x * char_width, y * char_height)
                    draw.text(pos, char, fill=color or '#ffffff', font=font)
        
        return img
        
    def enhance_image(
        self,
        image: Image.Image,
        audio_features: Optional[Dict] = None
    ) -> Image.Image:
        """Enhance image through synthesis pipeline."""
        # Prepare prompt with audio-reactive modifications
        prompt = self.style_prompt
        if audio_features:
            # Modify prompt based on audio features
            if audio_features.get('is_beat', False):
                prompt += ", pulsating energy"
            if audio_features.get('spectral_centroid', 0) > 0.7:
                prompt += ", bright, luminous"
            elif audio_features.get('spectral_centroid', 0) < 0.3:
                prompt += ", dark, mysterious"
        
        # Image-to-image synthesis
        result = self.img2img(
            prompt=prompt,
            image=image,
            negative_prompt=self.negative_prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength
        ).images[0]
        
        # Upscale result
        result = self.upscaler(
            prompt=prompt,
            image=result,
            num_inference_steps=self.num_inference_steps // 2
        ).images[0]
        
        # Apply inpainting for details
        mask = self._generate_detail_mask(result)
        if mask.getcolors(1)[0][1] > 0:  # If mask has non-black pixels
            result = self.inpaint(
                prompt=prompt + ", intricate details, high detail",
                image=result,
                mask_image=mask,
                num_inference_steps=self.num_inference_steps // 2
            ).images[0]
        
        return result
        
    def _generate_detail_mask(self, image: Image.Image) -> Image.Image:
        """Generate mask for areas needing detail enhancement."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Edge detection
        edges = cv2.Canny(img_array, 100, 200)
        
        # Dilate edges
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=2)
        
        return Image.fromarray(mask)
        
    def interpolate_frames(
        self,
        frame1: Image.Image,
        frame2: Image.Image,
        steps: int = 5
    ) -> List[Image.Image]:
        """Generate smooth transitions between frames."""
        # Convert images to numpy arrays
        img1 = np.array(frame1)
        img2 = np.array(frame2)
        
        # Calculate optical flow
        flow = self.frame_interpolator.calc(
            cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY),
            None
        )
        
        frames = []
        for i in range(steps):
            # Calculate interpolation factor
            t = (i + 1) / (steps + 1)
            
            # Warp first image towards second
            h, w = flow.shape[:2]
            flow_map = np.column_stack((
                np.indices((h, w)).transpose(1, 2, 0),
                np.ones((h, w))
            ))
            flow_map = flow_map.reshape(-1, 3)
            
            # Apply flow
            warped = cv2.remap(
                img1,
                (flow_map[:, 1] + flow[:, :, 0] * t).reshape(h, w),
                (flow_map[:, 0] + flow[:, :, 1] * t).reshape(h, w),
                cv2.INTER_LINEAR
            )
            
            # Blend with second image
            frame = cv2.addWeighted(warped, 1 - t, img2, t, 0)
            frames.append(Image.fromarray(frame))
            
        return frames
        
    def render_frame(
        self,
        pattern: str,
        color: Optional[str] = None,
        audio_features: Optional[Dict] = None,
        interpolate: bool = True
    ) -> Union[Image.Image, List[Image.Image]]:
        """Render a complete frame with synthesis and interpolation."""
        # Convert ASCII to base image
        base_image = self.ascii_to_image(pattern, color)
        
        # Enhance through synthesis
        enhanced = self.enhance_image(base_image, audio_features)
        
        if not interpolate or self.last_frame is None:
            self.last_frame = enhanced
            return enhanced
            
        # Generate transition frames
        transitions = self.interpolate_frames(self.last_frame, enhanced)
        self.last_frame = enhanced
        
        return [self.last_frame] + transitions
        
    def render_video(
        self,
        patterns: List[str],
        colors: Optional[List[str]] = None,
        audio_features: Optional[List[Dict]] = None,
        output_path: str = "output.mp4"
    ):
        """Render a complete video from patterns."""
        if colors is None:
            colors = [None] * len(patterns)
        if audio_features is None:
            audio_features = [None] * len(patterns)
            
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            self.output_size
        )
        
        try:
            for pattern, color, features in zip(patterns, colors, audio_features):
                # Render frame(s)
                frames = self.render_frame(pattern, color, features, interpolate=True)
                if not isinstance(frames, list):
                    frames = [frames]
                    
                # Write frames
                for frame in frames:
                    # Convert PIL to OpenCV format
                    cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                    out.write(cv_frame)
                    
        finally:
            out.release()
            
    def cleanup(self):
        """Clean up resources."""
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache() 