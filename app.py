from authtoken import auth_token
import streamlit as st
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 

modelid = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)

prompt = st.text_area("ENTER YOUR CREATIVITY")

def generate(): 
    with autocast(torch.device(device)):
        image = pipe(prompt.get(), guidance_scale=9.5)["sample"][0]
    img = st.image(image)

button = st.button("GENERATE")

if button:
    generate()


