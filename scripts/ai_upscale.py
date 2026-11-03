from pathlib import Path
import cv2
import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "AI-1920x1080"
OUT.mkdir(exist_ok=True)
WEIGHTS = ROOT / "RealESRGAN_x4plus.pth"

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4,
)
upsampler = RealESRGANer(
    scale=4,
    model_path=str(WEIGHTS),
    model=model,
    tile=256,
    tile_pad=16,
    pre_pad=0,
    half=False,
    device=torch.device("cpu"),
)

sources = sorted(ROOT.glob("IMG_20261103_*.jpg"))
if len(sources) != 5:
    raise RuntimeError(f"Expected 5 original wallpapers, found {len(sources)}")

for source in sources:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {source}")
    restored, _ = upsampler.enhance(image, outscale=4)
    # Reduce the AI-upscaled image to desktop Full HD. This keeps the
    # original framing and avoids the previous crop/stretch operation.
    rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(rgb).resize((1920, 1080), Image.Resampling.LANCZOS)
    destination = OUT / source.name
    result.save(destination, "JPEG", quality=96, optimize=True, subsampling=0)
    print(f"AI-upscaled {source.name}: {image.shape[1]}x{image.shape[0]} -> 1920x1080")
