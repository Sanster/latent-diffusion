import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from tqdm import tqdm
import numpy as np
import torch

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
except:
    pass

from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import cv2


class CondStageModelEncodeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)


class CondStageModelDecodeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)


def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    # image = cv2.resize(image, None, fx=0.5, fy=0.5)

    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    # mask = cv2.resize(mask, None, fx=0.5, fy=0.5)

    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1 - mask) * image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)  # model LatentDiffusion

    cond_stage_model_encode = torch.jit.load("/Users/qing/code/github/latent-diffusion/cond_stage_model_encode.pt")
    cond_stage_model_decode = torch.jit.load("/Users/qing/code/github/latent-diffusion/cond_stage_model_decode.pt")

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        # cond_stage_model_input = torch.FloatTensor(1, 3, 512, 512)
        # cond_stage_model_wrapper = CondStageModelEncodeWrapper(model.cond_stage_model)
        # traced_cond_stage_model = torch.jit.trace(cond_stage_model_wrapper, cond_stage_model_input)
        # torch.jit.save(traced_cond_stage_model, "cond_stage_model_encode.pt")

        # cond_stage_model_input = torch.FloatTensor(1, 3, 128, 128)
        # cond_stage_model_wrapper = CondStageModelDecodeWrapper(model.cond_stage_model)
        # traced_cond_stage_model = torch.jit.trace(cond_stage_model_wrapper, cond_stage_model_input)
        # torch.jit.save(traced_cond_stage_model, "cond_stage_model_decode.pt")

        for image, mask in tqdm(zip(images, masks)):
            outpath = os.path.join(opt.outdir, os.path.split(image)[1])
            # image [1,3,512,512] float32 mask: [1,1,512,512] float32 masked_image: [1,3,512,512] float32
            batch = make_batch(image, mask, device=device)

            # encode masked image and concat downsampled mask
            # c = model.cond_stage_model.encode(batch["masked_image"])  # 1,3,128,128
            c = cond_stage_model_encode(batch["masked_image"])

            cc = torch.nn.functional.interpolate(batch["mask"], size=c.shape[-2:])  # 1,1,128,128
            c = torch.cat((c, cc), dim=1)  # 1,4,128,128

            shape = (c.shape[1] - 1,) + c.shape[2:]
            samples_ddim = sampler.sample(steps=opt.steps,
                                          conditioning=c,
                                          batch_size=c.shape[0],
                                          shape=shape)

            x_samples_ddim = cond_stage_model_decode(samples_ddim)  # samples_ddim: 1, 3, 128, 128 float32

            image = torch.clamp((batch["image"] + 1.0) / 2.0,
                                min=0.0, max=1.0)
            mask = torch.clamp((batch["mask"] + 1.0) / 2.0,
                               min=0.0, max=1.0)
            predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                          min=0.0, max=1.0)

            inpainted = (1 - mask) * image + mask * predicted_image
            inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
            np_img = inpainted.astype(np.uint8)
            # np_img = cv2.resize(np_img, None, fx=2.0, fy=2.0)
            Image.fromarray(np_img).save(outpath)
