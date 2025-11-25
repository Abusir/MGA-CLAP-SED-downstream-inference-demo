# MGA-CLAP-SED-downstream-inference-demo
This is an unofficial implementation of inference code for the downstream SED task. The paper is '**Advancing Multi-grained Alignment for Contrastive Language-Audio Pre-training**', and the official implementation is https://github.com/Ming-er/MGA-CLAP

This is an extended version of `example.py` from the official repository [https://github.com/Ming-er/MGA-CLAP], taking into account the temperature parameter used during training and adding visualization function. Using `example.wav` from the official repository, you can obtain the following images.

![Example SED visualization](example_wav_sed.png)

One point to note is that, following the inference process of `MS-CLAP` [https://github.com/microsoft/CLAP] in ESC-50 dataset, I used `softmax` to calculate class probabilities. Therefore, in cases of multiple overlapping sounds, there may be issues with inaccurate class estimation, which is consistent with the original text.
