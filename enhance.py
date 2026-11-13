#---------
# IMPORTS
#---------
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#---------
# CONFIG
#---------
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
MODELS = {
    "photo": Path("weights/realesr-general-x4v3.pth"),
    "anime": Path("weights/realesr-animevideov3.pth"),
}
MODEL_NAME = os.environ.get("MODEL", "anime")
OVERWRITE = os.environ.get("OVERWRITE", "false").lower() == "true"
SMOOTH_LINES = os.environ.get("SMOOTH_LINES", "true").lower() == "true" and MODEL_NAME == "anime"
DARK_THRESHOLD = 60
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
NATIVE_SCALE = 4
AI_X4_MAX_MP = 2.0
AI_X2_MAX_MP = 6.0
MAX_SIDE = 4096
TILE = 200
TILE_PAD = 10
JPEG_QUALITY = 95

#---------
# MODEL
#---------
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(3, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def load_model():
    state = torch.load(MODELS[MODEL_NAME], map_location="cpu")
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    conv_count = sum(1 for k, v in state.items() if k.endswith(".weight") and v.ndim == 4)
    model = SRVGGNetCompact(num_conv=conv_count - 2)
    model.load_state_dict(state)
    model.eval()
    return model

#---------
# AI UPSCALE
#---------
def ai_upscale(model, img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    h, w = tensor.shape[2:]
    s = NATIVE_SCALE
    out = torch.zeros(1, 3, h * s, w * s)
    with torch.no_grad():
        for y in range(0, h, TILE):
            for x in range(0, w, TILE):
                y0, y1 = max(y - TILE_PAD, 0), min(y + TILE + TILE_PAD, h)
                x0, x1 = max(x - TILE_PAD, 0), min(x + TILE + TILE_PAD, w)
                result = model(tensor[:, :, y0:y1, x0:x1])
                th, tw = min(TILE, h - y), min(TILE, w - x)
                oy, ox = (y - y0) * s, (x - x0) * s
                out[:, :, y * s:(y + th) * s, x * s:(x + tw) * s] = result[:, :, oy:oy + th * s, ox:ox + tw * s]
    arr = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

#---------
# CLASSIC FALLBACK
#---------
def classic_enhance(img):
    img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 11)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

#---------
# ALPHA
#---------
def split_alpha(img):
    if img.dtype != np.uint8:
        img = (img / 257.0).astype(np.uint8) if img.dtype == np.uint16 else img.astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None
    if img.shape[2] == 4:
        bgr = np.ascontiguousarray(img[:, :, :3])
        alpha = np.ascontiguousarray(img[:, :, 3])
        hidden = (alpha == 0).astype(np.uint8) * 255
        if hidden.any() and not hidden.all():
            bgr = cv2.inpaint(bgr, hidden, 3, cv2.INPAINT_TELEA)
        return bgr, alpha
    return img, None


def resize_alpha(alpha, shape):
    if alpha is None:
        return None
    h, w = shape[:2]
    return cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC)


def merge_alpha(bgr, alpha):
    if alpha is None:
        return bgr
    return np.dstack((bgr, alpha))

#---------
# LINE SMOOTHING
#---------
def smooth_alpha(alpha, sigma):
    blurred = cv2.GaussianBlur(alpha, (0, 0), sigma).astype(np.float32)
    return np.clip((blurred - 127.5) * 3.0 + 127.5, 0, 255).astype(np.uint8)


def smooth_lines(bgr, alpha, scale):
    sigma = max(1.2, 0.6 * scale)
    brightest = bgr.max(axis=2)
    inside = np.ones(brightest.shape, dtype=bool) if alpha is None else alpha > 8
    dark = (brightest < DARK_THRESHOLD) & inside
    if not dark.any():
        return bgr, alpha
    source = (dark | ~inside).astype(np.float32)
    smooth = (cv2.GaussianBlur(source, (0, 0), sigma) > 0.5) & inside
    ink = np.median(bgr[dark], axis=0).astype(np.float32)
    hole = cv2.dilate(dark.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    filled = cv2.inpaint(bgr, hole, 3, cv2.INPAINT_TELEA)
    soft = cv2.GaussianBlur(smooth.astype(np.float32), (0, 0), 0.8)[..., None]
    out = filled.astype(np.float32) * (1.0 - soft) + ink * soft
    out = np.clip(out.round(), 0, 255).astype(np.uint8)
    return out, (smooth_alpha(alpha, sigma) if alpha is not None else None)

#---------
# PIPELINE
#---------
def limit_side(img):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= MAX_SIDE:
        return img
    k = MAX_SIDE / longest
    return cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)


def enhance(model, img):
    h, w = img.shape[:2]
    mp = h * w / 1e6
    if mp <= AI_X4_MAX_MP:
        mode = "AI x4"
        result = ai_upscale(model, img)
    elif mp <= AI_X2_MAX_MP:
        mode = "AI x2"
        small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        result = ai_upscale(model, small)
    else:
        mode = "classic"
        result = classic_enhance(img)
    result = limit_side(result)
    return result, mode, result.shape[1] / w

#---------
# IO
#---------
def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def write_image(path, img):
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if ext in {".jpg", ".jpeg"} else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(str(path))
    return ok

#---------
# MAIN
#---------
def main():
    torch.set_num_threads(os.cpu_count() or 2)
    OUTPUT_DIR.mkdir(exist_ok=True)
    files = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in EXTENSIONS)
    todo = [p for p in files if OVERWRITE or not (OUTPUT_DIR / p.name).exists()]
    if not todo:
        print("Nothing to process")
        return
    model = load_model()
    print(f"model: {MODEL_NAME}")
    for src in todo:
        started = time.time()
        img = read_image(src)
        if img is None:
            print(f"cannot read {src.name}")
            continue
        bgr, alpha = split_alpha(img)
        result, mode, scale = enhance(model, bgr)
        alpha = resize_alpha(alpha, result.shape)
        if SMOOTH_LINES and mode != "classic":
            result, alpha = smooth_lines(result, alpha, scale)
        result = merge_alpha(result, alpha)
        write_image(OUTPUT_DIR / src.name, result)
        print(f"{src.name}: {mode}, {bgr.shape[1]}x{bgr.shape[0]} -> {result.shape[1]}x{result.shape[0]}, {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
