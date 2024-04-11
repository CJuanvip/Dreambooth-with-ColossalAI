# [DreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) by [colossalai](https://github.com/hpcaitech/ColossalAI.git)

This repo is to reproduce [DreamBooth](https://arxiv.org/abs/2208.12242) in [ColossalAI](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth). DreamBooth is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.
The `train_dreambooth_colossalai.py` script shows how to implement the training procedure and adapt it for stable diffusion.

By accommodating model data in CPU and GPU and moving the data to the computing device when necessary, [Gemini](https://www.colossalai.org/docs/advanced_tutorials/meet_gemini), the Heterogeneous Memory Manager of [Colossal-AI](https://github.com/hpcaitech/ColossalAI) can breakthrough the GPU memory wall by using GPU and CPU memory (composed of CPU DRAM or nvme SSD memory) together at the same time. Moreover, the model scale can be further improved by combining heterogeneous training with the other parallel approaches, such as data parallel, tensor parallel and pipeline parallel.

## Installation

To reproduce without error, please use python <= 3.9 to avoid OS Error in [transformers](https://huggingface.co/docs/transformers/index) package, and make sure to install the library's training dependencies:

```bash
pip install -r requirements.txt
```

### Install [colossalai](https://github.com/hpcaitech/ColossalAI.git)

```bash
pip install colossalai
```

## Dataset used for reference images
Dreambooth is a subject-driven generation methods. In this repo, we use 5 photos of a cat named Maomao under the folder `/data`.

## Training

Simply run:

```bash
bash dreambooth_colossalai.sh
```

## Inference

Once the personal model is trained, simply run:

```bash
python inference.py
```

## Experiments Results & Handouts

The experiments log is at the path `output/rank_0.log`. The subject-driven generation is at `output/output.png`.


