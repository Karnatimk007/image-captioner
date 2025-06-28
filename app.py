import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

# Load model & processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Gradio interface
demo = gr.Interface(fn=caption_image, inputs=gr.Image(type="pil"), outputs="text", title="🖼️ Image Captioner")
demo.launch()
