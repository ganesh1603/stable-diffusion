from PIL import ImageTk
from authtoken import auth_token
import streamlit as st
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 



modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16",torch_dtype=torch.float16 ,use_auth_token=auth_token) 
pipe.to(device)


prompt=st.text_area("ENTER YOUR CREATIVITY")

def generate(): 
    with autocast(device): 
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    img=st.image(image)



button=st.button("GENERATE")

if button=="GENERATE":
    generate()
else:
    pass

