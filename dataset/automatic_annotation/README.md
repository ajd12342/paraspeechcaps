# Automatic Annotation Pipeline

This directory contains the code for automatically annotating speech utterances with style tags, as described in Sections 3 and 4 of the paper.

The automatic annotation pipeline has three parts:

## 1. [Basic Tags](./basic_tags/)
Extract fundamental acoustic and speaker attributes: **gender**, **pitch**, **speaking rate**, and **noise level**. (Paper Section 3.2)

## 2. [Intrinsic Tag Scaling](./intrinsic/) *(coming soon)*
Scale speaker-level intrinsic tags (e.g., *shrill*, *guttural*, *breathy*) from human-annotated speakers to new speakers using VoxSim perceptual speaker similarity matching. (Paper Section 4.1)

## 3. [Situational Tag Scaling](./situational/) *(coming soon)*
Scale utterance-level situational tags (e.g., *happy*, *whispered*, *sarcastic*) using a 3-step pipeline: DVA expressivity filtering, SFR-Embedding-Mistral semantic matching, and Gemini 1.5 Flash acoustic verification. (Paper Section 4.2)

After extracting all tags, the final step is to generate natural language style descriptions using an LLM — see [`../style_prompts/`](../style_prompts/) *(coming soon)*.
