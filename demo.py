from numpy.core.fromnumeric import _resize_dispatcher
import torch
from models.vgg16_drnet import vgg16dres
import gdown
from PIL import Image
from torchvision import transforms
import gradio as gr
import cv2
import numpy as np
import scipy


sha_path = "pretrained_models/dr_sha.pth"
shb_path = "pretrained_models/dr_shb.pth"
ucf_path = "pretrained_models/dr_ucf.pth"

# url = "https://drive.google.com/uc?id=1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru"
# gdown.download(url, model_path, quiet=False)


def load_return_model(path, model):
    model.load_state_dict(torch.load(path, device))
    model.eval()
    return model


device = torch.device('cuda')  # device can be "cpu" or "gpu"

models = {
    'ucf': None,  # load_return_model(ucf_path, vgg16dres(map_location=device)), 
    'sha': load_return_model(sha_path, vgg16dres(map_location=device).to(device)),
    'shb': None  # load_return_model(shb_path, vgg16dres(map_location=device)),
}


def predict(inp, model):
    
    inp = Image.fromarray(inp.astype('uint8'), 'RGB')
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    inp = inp.to(device)
    with torch.set_grad_enabled(False):
        outputs, _ = models[model](inp)
    count = torch.sum(outputs).item()
    vis_img = outputs[0, 0].cpu().numpy()
    # normalize density map values from 0 to 1, then map it to 0-255.
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    return vis_img, int(count)


title = "Distribution Matching for Crowd Counting"
desc = "A demo of DM-Count, a NeurIPS 2020 paper by Wang et al. Outperforms the state-of-the-art methods by a " \
       "large margin on four challenging crowd counting datasets: UCF-QNRF, NWPU, ShanghaiTech, and UCF-CC50. " \
       "This demo uses the QNRF trained model. Try it by uploading an image or clicking on an example " \
       "(could take up to 20s if running on CPU)."
examples = [
    ["/home/selman/Desktop/UCF_CC_50/16.jpg"],
]
inputs = [gr.inputs.Image(label="Image of Crowd", shape=(256, 256)),
          gr.inputs.Dropdown(choices=['sha', 'shb', 'ucf'], label='Trained Dataset')]
outputs = [gr.outputs.Image(label="Predicted Density Map"), gr.outputs.Label(label="Predicted Count")]
gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title=title, description=desc, examples=examples,
             allow_flagging=False).launch(share=True)
