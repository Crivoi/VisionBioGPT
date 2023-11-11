# VisionBioGPT
## Radiology report generation and clasification
### M.Sc. Thesis for the University of Copenhagen, September 2023
### Author: Andrei Crivoi

## Abstract 
With the emergence of the Transformer model (\cite{vaswani2017attention}), more and more engineers have presented different approaches in using this revolutionary architecture to solve and automate difficult medical tasks.

The task of disease classification is one such example. Doctors are usually tasked with assigning a diagnosis and execute different procedures to help them identify the issues of each patient. Within radiology, for instance, they use imaging support for disease classification, mostly represented as x-rays. This, however, is a very difficult and prone to error process that should not be taken lightly.

For this reason, we propose VisionBioGPT, a BioGPT (\cite{biogpt}) - Vision Transformer (\cite{vit}) hybrid, built on top of the Vision Encoder Decoder system (\cite{li2022trocr}, \cite{ramos2023smallcap}). We aim to evaluate the performance of BioGPT, a GPT-2 (\cite{radford2019gpt2}) based model pre-trained from scratch on large text corpus of biomedical data from PubMed\footnote{PubMed available at \href{https://pubmed.ncbi.nlm.nih.gov/}{pubmed.ncbi.nlm.nih.gov}}, when tackling radiology-specific tasks (i.e. report, x-ray disease classification and report generation).

We initially conduct some experiments to verify BioGPT's classification and generative capabilities when faced with reports from the \textbf{MIMIC-III} (\cite{Johnson2016MIMICIII}) dataset and compare our results to those of \cite{dai2022revisiting} on similar experiments using the same data, but different models. We establish that BioGPT achieves comparable performance to our point of reference and obtain new pre-trained weights that are more suitable in understanding radiology reports written in the \textbf{MIMIC} format. 

We then extend our experiments to a new dataset: \textbf{MIMIC-CXR} (\cite{Johnson2019MIMICCXR}). We first execute a classification task and obtain promising results with our task-adaptive pre-trained weights. Then, we alter BioGPT's attention block with a cross-attention layer and use it as text-decoder, together with the Vision Transformer as image-encoder and add chest x-rays to our input sequence. This method shows considerably reduced performance compared to our baseline text-only approach.

We believe that our experiments will entice researchers in further experimenting with the BioGPT model in text-only setups for disease classification and expand the model to newer GPT architectures (\cite{brown2020gpt3}, \cite{openai2023gpt4}).