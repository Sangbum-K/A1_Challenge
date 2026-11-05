"""Offline only: export unchanged checkpoint weights for the C++ runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
import timm

ROOT = Path(__file__).resolve().parents[2] / 'reference/A1_RealCar_A2_P5Brake'
sys.path.insert(0, str(ROOT / 'scripts'))
from live_a2_brake import load_backbone, load_a2

class Tokenizer(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, image):
        x = self.backbone(image)[-1]
        return torch.nn.functional.adaptive_avg_pool2d(x, (1, 40)).squeeze(2).transpose(1, 2)

class Head(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, tokens, ego):
        return self.model.forward_tokens([tokens[i] for i in range(21)], ego)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    device = torch.device(args.device)
    paths = [ROOT/'senior_code/data/ckpts'/n for n in
             ('seg_pretrain_fx640_v3_backbone.pt', 'a2_mf20_v2s_n20.pt')]
    backbone, binfo = load_backbone('tf_efficientnetv2_s', str(paths[0]), device, False)
    model, ainfo = load_a2('tf_efficientnetv2_s', str(paths[1]), device, False)
    tok, head = Tokenizer(backbone).eval(), Head(model).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    errors = []
    with torch.inference_mode():
        image = torch.randn(1, 3, 250, 1280, device=device)
        tokens = tok(image).unsqueeze(0).repeat(21, 1, 1, 1)
        ego = torch.zeros(1, 2, device=device)
        traced_tok = torch.jit.trace(tok, image, check_trace=False)
        traced_head = torch.jit.trace(head, (tokens, ego), check_trace=False)
        # Independent inputs, beyond the trace example, are checked numerically.
        for _ in range(3):
            x = torch.randn_like(image)
            expected = tok(x)
            torch.testing.assert_close(traced_tok(x), expected, rtol=1e-4, atol=1e-5)
            ts = torch.randn_like(tokens)
            ref, actual = head(ts, ego), traced_head(ts, ego)
            torch.testing.assert_close(actual, ref, rtol=1e-4, atol=1e-5)
            errors.append(float((actual-ref).abs().max()))
        traced_tok.save(str(args.out/'tokenizer.pt'))
        traced_head.save(str(args.out/'a2_head.pt'))
    info = dict(torch=torch.__version__, timm=timm.__version__, device=args.device,
                tf32=False, profiling_executor=False, precision='float32', backbone=binfo, a2=ainfo,
                head_max_abs_errors=errors, shape=[1,40,7], ego_motion=[0,0],
                checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (args.out/'export.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))

if __name__ == '__main__':
    main()
