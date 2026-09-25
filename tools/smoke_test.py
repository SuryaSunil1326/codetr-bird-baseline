import torch
import mmcv
from mmcv import Config
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.models import build_backbone

CONFIG = 'projects/configs/my_exps/co_dino_5scale_vit_large_bird_instance.py'

cfg = Config.fromfile(CONFIG)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch', torch.__version__, '| mmcv', mmcv.__version__, '| device', device)

backbone = build_backbone(cfg.model.backbone).to(device).eval()
x = torch.randn(1, 3, 512, 512, device=device)
with torch.no_grad():
    feats = backbone(x)

print('backbone output shapes:', [tuple(f.shape) for f in feats])
assert len(feats) > 0 and all(torch.isfinite(f).all() for f in feats)
print('smoke test ok')
