import torch
import numpy as np
from torchvision.transforms import ToTensor
import threading
import warnings

from networks.alac_gan import Colorizer as AlacGANGenerator
from networks.cycle_gan import CycleGANGenerator
from utils.utils import resize_pad, tile_process

# Enable CUDA optimizations for faster inference
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class ColorizationStrategy:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model = None
        # Check if FP16 should be used
        self.use_fp16 = getattr(config, 'use_fp16', False) and self.device == 'cuda' and torch.cuda.is_available()
        # Check if torch.compile should be used (PyTorch 2.0+)
        self.use_compile = getattr(config, 'use_compile', False) and hasattr(torch, 'compile')

    def load_weights(self, path):
        raise NotImplementedError

    def process_image(self, image, size):
        raise NotImplementedError


class AlacGANStrategy(ColorizationStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.model = AlacGANGenerator().to(self.device)
        self.load_weights(config.colorizer_path)
        self.model.eval()
        self.params = config.alacgan
        # torch.compile can be unstable with dynamic shapes and/or concurrent requests.
        # Keep both eager and compiled handles and allow a thread-safe global fallback.
        self._lock = threading.Lock()
        self._eager_model = self.model
        self._compiled_model = None
        
        # Convert to half precision if enabled
        if self.use_fp16:
            try:
                self.model = self.model.half()
                print(f"[+] AlacGAN converted to FP16 for faster inference")
            except Exception as e:
                print(f"[-] Failed to convert AlacGAN to FP16: {e}")
                self.use_fp16 = False
        
        # Compile model for PyTorch 2.0+ if enabled
        if self.use_compile:
            try:
                self._compiled_model = torch.compile(self.model, mode='reduce-overhead')
                self.model = self._compiled_model
                print(f"[+] AlacGAN compiled with torch.compile()")
            except Exception as e:
                print(f"[-] Failed to compile AlacGAN: {e}")
                self.use_compile = False
                self._compiled_model = None
                self.model = self._eager_model

    def load_weights(self, path):
        try:
            # Suppress SourceChangeWarning when loading models saved with different PyTorch versions
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*source code of class.*has changed.*')
            state_dict = torch.load(path, map_location=self.device)
            if 'module' in list(state_dict.keys())[0]:
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.generator.load_state_dict(state_dict)
            print(f"[+] Loaded AlacGAN weights from {path}")
        except Exception as e:
            print(f"[-] Failed to load AlacGAN weights: {e}")

    def process_image(self, image, size):
        target_size = size if size > 0 else self.params.image_size
        if target_size % 32 != 0: target_size = (target_size // 32) * 32

        processed_img, pad = resize_pad(image, target_size)
        img_tensor = ToTensor()(processed_img).unsqueeze(0).to(self.device)
        hint = torch.zeros(1, 4, img_tensor.shape[2], img_tensor.shape[3]).to(self.device)
        
        # Use appropriate dtype
        if self.use_fp16:
            img_tensor = img_tensor.half()
            hint = hint.half()
        else:
            img_tensor = img_tensor.float()
            hint = hint.float()

        with torch.no_grad():
            model_input = torch.cat([img_tensor, hint], 1)

            if self.params.tile_size > 0:
                try:
                    fake_color = tile_process(
                        self.model, model_input, 1,
                        self.params.tile_size, self.params.tile_pad
                    )
                except Exception as e:
                    # If compiled path fails (common with inductor/cudagraphs), disable it globally and retry once.
                    if self.use_compile and self._compiled_model is not None:
                        with self._lock:
                            if self.use_compile and self._compiled_model is not None:
                                print(f"[-] AlacGAN compiled path failed ({type(e).__name__}: {e}). Disabling compile and falling back to eager.")
                                self.use_compile = False
                                self._compiled_model = None
                                self.model = self._eager_model
                        fake_color = tile_process(
                            self.model, model_input, 1,
                            self.params.tile_size, self.params.tile_pad
                        )
                    else:
                        raise
            else:
                try:
                    fake_color, _ = self.model(model_input)
                except Exception as e:
                    if self.use_compile and self._compiled_model is not None:
                        with self._lock:
                            if self.use_compile and self._compiled_model is not None:
                                print(f"[-] AlacGAN compiled path failed ({type(e).__name__}: {e}). Disabling compile and falling back to eager.")
                                self.use_compile = False
                                self._compiled_model = None
                                self.model = self._eager_model
                        fake_color, _ = self.model(model_input)
                    else:
                        raise

            result = fake_color[0].detach().permute(1, 2, 0) * 0.5 + 0.5
            if pad[0] != 0: result = result[:-pad[0]]
            if pad[1] != 0: result = result[:, :-pad[1]]

        return (result.float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


class CycleGANStrategy(ColorizationStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.model = CycleGANGenerator(input_nc=3, output_nc=3, ngf=64).to(self.device)
        self.load_weights(config.colorizer_path)
        self.model.eval()  # Changed from train() to eval() for faster inference
        self.params = config.cyclegan
        self._lock = threading.Lock()
        self._eager_model = self.model
        self._compiled_model = None
        
        # Convert to half precision if enabled
        if self.use_fp16:
            try:
                self.model = self.model.half()
                print(f"[+] CycleGAN converted to FP16 for faster inference")
            except Exception as e:
                print(f"[-] Failed to convert CycleGAN to FP16: {e}")
                self.use_fp16 = False
        
        # Compile model for PyTorch 2.0+ if enabled
        if self.use_compile:
            try:
                self._compiled_model = torch.compile(self.model, mode='reduce-overhead')
                self.model = self._compiled_model
                print(f"[+] CycleGAN compiled with torch.compile()")
            except Exception as e:
                print(f"[-] Failed to compile CycleGAN: {e}")
                self.use_compile = False
                self._compiled_model = None
                self.model = self._eager_model

    def load_weights(self, path):
        try:
            # Suppress SourceChangeWarning when loading models saved with different PyTorch versions
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*SourceChangeWarning.*')
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                sd = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                sd = checkpoint['model']
            elif isinstance(checkpoint, dict):
                sd = checkpoint
            else:
                sd = checkpoint.state_dict()
            self.model.load_state_dict(sd, strict=True)
            print(f"[+] Loaded CycleGAN weights from {path}")
        except Exception as e:
            print(f"[-] Failed to load CycleGAN weights: {e}")

    def process_image(self, image, size):
        target_size = size if size > 0 else self.params.image_size
        if target_size % 4 != 0: target_size = (target_size // 4) * 4

        processed_img, pad = resize_pad(image, target_size)
        img_tensor = ToTensor()(processed_img).unsqueeze(0).to(self.device)
        img_tensor = (img_tensor - 0.5) / 0.5
        
        # Use appropriate dtype
        if self.use_fp16:
            img_tensor = img_tensor.half()
        else:
            img_tensor = img_tensor.float()

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat(1, 3, 1, 1)

        with torch.no_grad():
            if self.params.tile_size > 0:
                try:
                    fake_color = tile_process(
                        self.model, img_tensor, 1,
                        self.params.tile_size, self.params.tile_pad
                    )
                except Exception as e:
                    if self.use_compile and self._compiled_model is not None:
                        with self._lock:
                            if self.use_compile and self._compiled_model is not None:
                                print(f"[-] CycleGAN compiled path failed ({type(e).__name__}: {e}). Disabling compile and falling back to eager.")
                                self.use_compile = False
                                self._compiled_model = None
                                self.model = self._eager_model
                        fake_color = tile_process(
                            self.model, img_tensor, 1,
                            self.params.tile_size, self.params.tile_pad
                        )
                    else:
                        raise
            else:
                try:
                    fake_color = self.model(img_tensor)
                except Exception as e:
                    if self.use_compile and self._compiled_model is not None:
                        with self._lock:
                            if self.use_compile and self._compiled_model is not None:
                                print(f"[-] CycleGAN compiled path failed ({type(e).__name__}: {e}). Disabling compile and falling back to eager.")
                                self.use_compile = False
                                self._compiled_model = None
                                self.model = self._eager_model
                        fake_color = self.model(img_tensor)
                    else:
                        raise

            result = fake_color[0].detach().permute(1, 2, 0) * 0.5 + 0.5
            if pad[0] != 0: result = result[:-pad[0]]
            if pad[1] != 0: result = result[:, :-pad[1]]

        return (result.float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


class MangaColorizator:
    def __init__(self, config):
        if config.device == 'cuda' and not torch.cuda.is_available():
            print("[-] CUDA not available, using CPU.")
            config.device = 'cpu'

        self.config = config
        self.current_image = None
        self.current_size = 576

        if config.colorizer_type == 'CycleGAN':
            self.strategy = CycleGANStrategy(config)
        elif config.colorizer_type == 'AlacGAN':
            self.strategy = AlacGANStrategy(config)
        else:
            raise Exception('Invalid colorizer type')

    def set_image(self, image, size=0):
        self.current_image = image
        self.current_size = size

    def colorize(self):
        if self.current_image is None:
            raise RuntimeError("Image not set. Call set_image() first.")

        return self.strategy.process_image(self.current_image, self.current_size)