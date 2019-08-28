import os
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision as tv
from tqdm import tqdm, trange

from config import img_height, img_width, num_classes, root_dir, model_dir, optim_dir
from model import FastSCN
from utils import semantic_to_rgb, to_pil, to_tensor

visualize = True
retrain = True
batch_size = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FastSCN(img_height, img_width, num_classes).to(device)
optim = torch.optim.Adam(model.parameters(), lr=.001)
criterion = torch.nn.NLLLoss()

if visualize:
    import visdom
    viz = visdom.Visdom("http://localhost", env="Fast-SCN")
    loss_win = "loss win"
    img_win = "img win"
    gt_win = "ground truth win"
    pred_win = "pred win"

if retrain:
    if os.path.exists(model_dir):
        state_dict = torch.load(model_dir, map_location=device)
        model.load_state_dict(state_dict)
        del state_dict
    else:
        raise FileNotFoundError("Model not found for retraining.")
    if os.path.exists(optim_dir):
        state_dict = torch.load(optim_dir, map_location=device)
        optim.load_state_dict(state_dict)
        del state_dict
    else:
        raise FileNotFoundError("Optimizer not found for retraining.")

transforms = tv.transforms.Compose([
    tv.transforms.ColorJitter(saturation=.05, hue=.05), 
    tv.transforms.ToTensor(), 
    tv.transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
])
target_transforms = tv.transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
ds = tv.datasets.Cityscapes(
    root_dir, split="val", mode="fine", target_type="semantic", 
    transform=transforms, target_transform=target_transforms
)
data_loader = torch.utils.data.DataLoader(
    ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
)

num_epochs = 0
for epoch in range(num_epochs):
    model.train()
    losses = []
    for img, lbl in tqdm(data_loader):
        img = img.to(device)
        lbl = lbl.to(device)
        optim.zero_grad()
        pred = model(img)
        loss = criterion(pred, lbl)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        del img, lbl, pred, loss
    else:
        print("################################################################")
        print("Epoch %d completed."%(epoch+1))
        print("Mean loss: ", np.mean(losses))
        print("################################################################")
        #Save the model
        torch.save(model.state_dict(), model_dir)
        torch.save(optim.state_dict(), optim_dir)
        if visualize:
            #Visualize the losses
            viz.line(
                np.array(losses), np.arange(epoch*len(data_loader), epoch*len(data_loader)+len(data_loader)), 
                update="append", win=loss_win, opts={"title": "Loss", "legend": ["Training loss"]}
            )
            #Generate a sample and visualize it
            model.eval()
            ind = np.random.randint(low=0, high=len(ds))
            img, lbl = ds[ind]
            img = img.unsqueeze(0).to(device)
            lbl = lbl.to(device)
            lbl = semantic_to_rgb(lbl.cpu().numpy())
            lbl = to_tensor(lbl)
            pred = model(img).squeeze(0)
            pred = torch.argmax(pred, dim=0)
            pred = semantic_to_rgb(pred.detach().cpu().numpy())
            pred = to_tensor(pred)
            img = img.squeeze(0) * .5 + .5
            viz.image(
                img.cpu().numpy(), win=img_win, 
                opts={"title": "Sample image", "height": 512, "width": 1024}
            )
            viz.image(
                lbl.numpy(), win=gt_win, 
                opts={"title": "Ground truth segmentation", "height": 512, "width": 1024}
            )
            viz.image(
                pred.numpy(), win=pred_win, 
                opts={"title": "Predicted segmentation", "height": 512, "width": 1024}
            )

if visualize:
    model.eval()
    ind = np.random.randint(low=0, high=len(ds))
    img, lbl = ds[ind]
    img = img.unsqueeze(0).to(device)
    lbl = lbl.to(device)
    pred = model(img).squeeze(0)
    pred = torch.argmax(pred, dim=0)
    pred = semantic_to_rgb(pred.detach().cpu().numpy())
    pred = to_tensor(pred)
    lbl = semantic_to_rgb(lbl.cpu().numpy())
    lbl = to_tensor(lbl)
    img = img.squeeze(0) * .5 + .5
    viz.image(
        img.cpu().numpy(), win=img_win, 
        opts={"title": "Sample image", "height": 512, "width": 1024}
    )
    viz.image(
        lbl.numpy(), win=gt_win, 
        opts={"title": "Ground truth segmentation", "height": 512, "width": 1024}
    )
    viz.image(
        pred.numpy(), win=pred_win, 
        opts={"title": "Predicted segmentation", "height": 512, "width": 1024}
    )
    viz.save(envs=["Fast-SCN"])
