import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel

model_id = "CompVis/stable-diffusion-v1-4"
print(f"Loading model... from{model_id}")

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# customized model
unet = UNet2DConditionModel.from_pretrained("output", torch_dtype=torch.float16).to("cuda")
pipe.unet = unet

prompt = "A photo of a qwertyuiop cat."
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("output/output.png")
