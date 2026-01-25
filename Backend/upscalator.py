import numpy as np
import torch
import threading
import warnings

from networks.RRDBNet import Upscaler as ESRGANNet
from networks.aura_sr import Upscaler as GigaGANNet, upscale_4x, upscale_4x_overlapped
from utils.utils import tile_process

# Enable CUDA optimizations for faster inference
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class UpscalingStrategy:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model = None
        # Check if FP16 should be used
        self.use_fp16 = getattr(config, 'use_fp16', False) and self.device == 'cuda' and torch.cuda.is_available()
        # Check if torch.compile should be used (PyTorch 2.0+)
        self.use_compile = getattr(config, 'use_compile', False) and hasattr(torch, 'compile')

    def upscale(self, image, factor):
        raise NotImplementedError


class ESRGANStrategy(UpscalingStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.model = ESRGANNet().to(self.device)
        self.load_weights(config.upscaler_path)
        self.model.eval()
        self.params = config.esrgan
        self._lock = threading.Lock()
        self._eager_model = self.model
        self._compiled_model = None
        
        # Convert to half precision if enabled
        if self.use_fp16:
            try:
                self.model = self.model.half()
                print(f"[+] ESRGAN converted to FP16 for faster inference")
            except Exception as e:
                print(f"[-] Failed to convert ESRGAN to FP16: {e}")
                self.use_fp16 = False
        
        # Compile model for PyTorch 2.0+ if enabled
        if self.use_compile:
            try:
                self._compiled_model = torch.compile(self.model, mode='reduce-overhead')
                self.model = self._compiled_model
                print(f"[+] ESRGAN compiled with torch.compile()")
            except Exception as e:
                print(f"[-] Failed to compile ESRGAN: {e}")
                self.use_compile = False
                self._compiled_model = None
                self.model = self._eager_model

    def load_weights(self, path):
        try:
            # Suppress SourceChangeWarning when loading models saved with different PyTorch versions
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*source code of class.*has changed.*')
            model_or_chkpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.generator = model_or_chkpt
            print(f"[+] Loaded ESRGAN weights  from {path}")
        except Exception as e:
            print(f"[-] Failed to load ESRGAN weights: {e}")

    def upscale(self, image, factor):
        img_tensor = torch.from_numpy(image).to(self.device)
        result = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Use appropriate dtype
        if self.use_fp16:
            result = result.half()
        else:
            result = result.float()

        with torch.no_grad():
            if self.params.tile_size > 0:
                try:
                    result = tile_process(
                        self.model,
                        result,
                        factor,
                        self.params.tile_size,
                        self.params.tile_pad
                    )
                except Exception as e:
                    if self.use_compile and self._compiled_model is not None:
                        with self._lock:
                            if self.use_compile and self._compiled_model is not None:
                                print(f"[-] ESRGAN compiled path failed ({type(e).__name__}: {e}). Disabling compile and falling back to eager.")
                                self.use_compile = False
                                self._compiled_model = None
                                self.model = self._eager_model
                        result = tile_process(
                            self.model,
                            result,
                            factor,
                            self.params.tile_size,
                            self.params.tile_pad
                        )
                    else:
                        raise
            else:
                try:
                    result = self.model(result)
                except Exception as e:
                    if self.use_compile and self._compiled_model is not None:
                        with self._lock:
                            if self.use_compile and self._compiled_model is not None:
                                print(f"[-] ESRGAN compiled path failed ({type(e).__name__}: {e}). Disabling compile and falling back to eager.")
                                self.use_compile = False
                                self._compiled_model = None
                                self.model = self._eager_model
                        result = self.model(result)
                    else:
                        raise

        result = result.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        result = np.transpose(result[[2, 1, 0], :, :], (1, 2, 0))
        result = (result * 255.0).round().astype(np.uint8)
        result = result[:, :, ::-1]

        return result


class GigaGANStrategy(UpscalingStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.model = GigaGANNet().to(self.device)
        self.load_weights(config.upscaler_path)
        self.model.eval()
        self.params = config.gigagan
        print(f"[*] GigaGAN Config: Batch={self.params.batch_size}, Overlap={self.params.use_overlap}")
        self._lock = threading.Lock()
        self._eager_model = self.model
        self._compiled_model = None
        
        # Convert to half precision if enabled
        if self.use_fp16:
            try:
                self.model = self.model.half()
                print(f"[+] GigaGAN converted to FP16 for faster inference")
            except Exception as e:
                print(f"[-] Failed to convert GigaGAN to FP16: {e}")
                self.use_fp16 = False
        
        # Compile model for PyTorch 2.0+ if enabled
        if self.use_compile:
            try:
                self._compiled_model = torch.compile(self.model, mode='reduce-overhead')
                self.model = self._compiled_model
                print(f"[+] GigaGAN compiled with torch.compile()")
            except Exception as e:
                print(f"[-] Failed to compile GigaGAN: {e}")
                self.use_compile = False
                self._compiled_model = None
                self.model = self._eager_model

    def load_weights(self, path):
        try:
            # Suppress SourceChangeWarning when loading models saved with different PyTorch versions
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*source code of class.*has changed.*')
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            if 'state_dict' in checkpoint:
                sd = checkpoint['state_dict']
                new_sd = {k.replace('model.', '').replace('generator.', ''): v for k, v in sd.items()}
                self.model.generator.load_state_dict(new_sd, strict=False)
            else:
                self.model.generator.load_state_dict(checkpoint, strict=False)
            print(f"[+] GigaGAN weights loaded from {path}")
        except Exception as e:
            print(f"[-] Failed to load GigaGAN weights: {e}")

    def upscale(self, image, factor):
        if factor != 4:
            print(f"[-] Warning: GigaGAN is natively 4x. Requested {factor}x.")

        try:
            req_size = self.model.generator.input_image_size
        except AttributeError:
            req_size = 64

        with torch.no_grad():
            if self.params.use_overlap:
                result = upscale_4x_overlapped(
                    image,
                    self.model,
                    input_image_size=req_size,
                    max_batch_size=self.params.batch_size
                )
            else:
                result = upscale_4x(
                    image,
                    self.model,
                    input_image_size=req_size,
                    max_batch_size=self.params.batch_size
                )

        if result.dtype != np.uint8:
            result = (result).clip(0, 255).astype(np.uint8)

        return result

class MangaUpscaler:
    def __init__(self, config):
        if config.device == 'cuda' and not torch.cuda.is_available():
            print("[-] CUDA not available, using CPU.")
            config.device = 'cpu'

        if config.upscaler_type == 'GigaGAN':
            self.strategy = GigaGANStrategy(config)
        elif config.upscaler_type == 'ESRGAN':
            self.strategy = ESRGANStrategy(config)
        else:
            raise Exception('Invalid upscaler type')

    def upscale(self, image, factor):
        if image.shape[2] == 4:
            image = image[:, :, :3]

        return self.strategy.upscale(image, factor)