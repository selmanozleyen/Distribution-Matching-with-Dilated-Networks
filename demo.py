import torch
from models.vgg16_drnet import vgg16dres
from models.vgg16_drnet1 import vgg16dres1
from models.vgg19 import vgg19
from PIL import Image
from torchvision import transforms
import gradio as gr
import cv2
import numpy as np


sha_path = "pretrained_models/dr_sha.pth"
shb_path = "pretrained_models/dilres_shb.pth"
ucf_path = "pretrained_models/dr_ucf.pth"
dm_shb_path = "pretrained_models/model_sh_B.pth"
dm_sha_path = "pretrained_models/model_sh_A.pth"

# url = "https://drive.google.com/uc?id=1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru"
# gdown.download(url, model_path, quiet=False)


def load_return_model(path, model):
    model.load_state_dict(torch.load(path, device))
    model.eval()
    return model


device = torch.device('cuda')  # device can be "cpu" or "gpu"

models = {
    'sha': load_return_model(sha_path, vgg16dres(map_location=device).to(device)),
    'ucf': load_return_model(ucf_path, vgg16dres1(map_location=device).to(device)),
    'shb': load_return_model(shb_path, vgg16dres1(map_location=device).to(device)),
    'dm_shb': load_return_model(dm_shb_path, vgg19().to(device)),
    'dm_sha': load_return_model(dm_sha_path, vgg19().to(device)),
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
    ["examples/ucf/13.jpg"],
]
inputs = [gr.inputs.Image(label="Image of Crowd"),
          gr.inputs.Dropdown(choices=['sha', 'shb', 'ucf', 'dm_shb'], label='Trained Dataset')]
outputs = [gr.outputs.Image(label="Predicted Density Map"), gr.outputs.Label(label="Predicted Count")]
gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title=title, description=desc, examples=examples,
             allow_flagging=False, live=False, allow_screenshot=False).launch(share=True)
