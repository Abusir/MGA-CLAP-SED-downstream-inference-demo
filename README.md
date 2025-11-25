# MGA-CLAP-SED-downstream-inference-demo

This repository provides an **unofficial implementation** of inference code for the downstream **Sound Event Detection (SED)** task based on the paper  
> **Advancing Multi-grained Alignment for Contrastive Language-Audio Pre-training**  
> [Official repository: https://github.com/Ming-er/MGA-CLAP](https://github.com/Ming-er/MGA-CLAP)

---

## Overview

This project extends the original `example.py` provided in the official repository by:
- Incorporating the **temperature parameter** used during training into the inference process.  
- Adding **visualization utilities** to display frame-level class probabilities as a time–class heatmap.

Using the `example.wav` file from the official repository, you can reproduce results similar to the figure below:

![Example SED visualization](example_wav_sed.png)

---

## Notes

This implementation **follows the zero-shot inference strategy** used in [Microsoft CLAP](https://github.com/microsoft/CLAP) on the **ESC-50 dataset**.  
Specifically, the model uses a `softmax` function over class similarities to compute class probabilities.  

As a result, when multiple sound events overlap within the same audio segment, the probabilities are mutually normalized, which may lead to **inaccurate class estimation for weaker events**—a limitation consistent with both the original MGA-CLAP paper.
