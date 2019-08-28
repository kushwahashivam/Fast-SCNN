import os
import time
from PIL import Image
import numpy as np
import torch
import torchvision as tv
from tqdm import tqdm, trange
import visdom

from config import img_height, img_width, num_classes, root_dir, model_dir
from model import FastSCN
from utils import semantic_to_rgb_numpy, to_pil, to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FastSCN(img_height, img_width, num_classes).to(device)

viz = visdom.Visdom("http://localhost", env="Fast-SCN")
loss_win = "loss win"
img_win = "img win"
gt_win = "ground truth win"
pred_win = "pred win"

if os.path.exists(model_dir):
    state_dict = torch.load(model_dir, map_location=device)
    model.load_state_dict(state_dict)
    del state_dict
else:
    raise FileNotFoundError("Model not found for retraining.")

transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(), 
    tv.transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
])
target_transforms = tv.transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
lbl_transform = tv.transforms.ToTensor()
ds = tv.datasets.Cityscapes(
    root_dir, split="val", mode="fine", target_type="semantic", 
    transform=transforms, target_transform=target_transforms
)

model.eval()
with torch.no_grad():
    ind = np.random.randint(low=0, high=len(ds))
    img, lbl = ds[ind]
    img = img.unsqueeze(0).to(device)
    lbl = semantic_to_rgb_numpy(lbl.numpy())
    lbl = lbl_transform(lbl).numpy()
    pred = model(img).squeeze(0)
    pred = torch.argmax(pred, dim=0)
    pred = semantic_to_rgb_numpy(pred.detach().cpu().numpy())
    pred = lbl_transform(pred).numpy()
    img = (img.squeeze(0) * .5 + .5).cpu().numpy()
    viz.image(
        img, win=img_win, 
        opts={"title": "Sample image", "height": 512, "width": 1024}
    )
    viz.image(
        lbl, win=gt_win, 
        opts={"title": "Ground truth segmentation", "height": 512, "width": 1024}
    )
    viz.image(
        pred, win=pred_win, 
        opts={"title": "Predicted segmentation", "height": 512, "width": 1024}
    )
viz.save(envs=["Fast-SCN"])