import os
import time
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision as tv

from config import img_height, img_width, num_classes, root_dir, model_dir, optim_dir
from model import FastSCN
from utils import semantic_to_rgb, to_pil, to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastSCN(img_height, img_width, num_classes).to(device).eval()
if os.path.exists(model_dir):
    state_dict = torch.load(model_dir, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
    del state_dict
else:
    raise FileNotFoundError("Model not found.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Unable to read camera feed.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
time.sleep(2)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Camera resolution: %d x %d"%(frame_width, frame_height))

transforms = tv.transforms.Compose([
    tv.transforms.CenterCrop(size=(frame_width//2, frame_width)), 
    tv.transforms.Resize(1024), 
    tv.transforms.ToTensor(), 
    tv.transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

with torch.no_grad():
    while True:
        init_t = time.time()
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transforms(img).unsqueeze(0).to(device)
        sem = model(img).squeeze(0)
        sem = torch.argmax(sem, dim=0)
        # sem = semantic_to_rgb(sem.detach().cpu().numpy())
        # sem = np.array(sem)
        # sem = cv2.cvtColor(sem, cv2.COLOR_RGB2BGR)
        # sem = cv2.resize(sem, dsize=(frame_width, frame_height))
        if ret == True:
            cv2.imshow("Camera input", frame)
            # cv2.imshow("Segmentation", sem)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("Could not read camera input.")
            break
        fin_t = time.time()
        print("FPS: ", 1/(fin_t-init_t))
cap.release()
cv2.destroyAllWindows()