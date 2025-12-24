# 🩻 MIMIC-CXR Radiology Report Generator

This repository provides the **codebase** for a **PyTorch encoder–decoder model** for automated radiology report generation using chest X-ray images and structured medical knowledge.

📌 **Pretrained model weights are hosted on Hugging Face:**  
https://huggingface.co/shemzegem/mimic-cxr-report-generator

📌 **Additional resources such as pickle file, adjacency matrix, model, grouped dataframe are in the following drive link:** 
https://drive.google.com/drive/folders/1DThVS3wzvL9EbdtBQjFHEWIO8ilqLrvz?usp=sharing

This GitHub repository focuses on:
- Model architecture and implementation
- Inference and evaluation code
- Training and experimentation notebooks

---

## 1. Overview

Radiology report generation is a challenging problem that requires the integration of:
- Accurate visual understanding of medical images
- Robust clinical language modeling
- Anatomical and pathological consistency

This project proposes a **hybrid multimodal architecture** that integrates:

1. Visual features extracted from dual-view chest X-rays  
2. Structured medical knowledge encoded using a Graph Convolutional Network (GCN)  
3. Sequential text generation using an LSTM-based decoder  

The model is trained and evaluated on the **MIMIC-CXR** dataset.

---

## 2. Model Architecture

### 2.1 Visual Encoder
- Backbone: ResNet-18
- Input: Two chest X-ray images (frontal + lateral)
- Output: Fixed-length image embeddings

Each image is processed independently, and the resulting embeddings are combined downstream.

---

### 2.2 Knowledge Graph Encoder
- Graph representation of medical entities
- Encoder: Graph Convolutional Network (GCN)
- Node features derived from textual embeddings
- Output: Global knowledge embedding via mean pooling

This component provides structured clinical priors that guide report generation.

---

### 2.3 Multimodal Fusion

The image embeddings and the knowledge graph embedding are concatenated and projected into a unified latent space:

```
[Image_1 | Image_2 | Knowledge_Graph] -> Linear_Projection -> Shared_Embedding
```

---

### 2.4 Report Decoder
- Decoder: LSTM-based RNN
- Input: Multimodal embedding + tokenized report prefix
- Output: Token-level probability distribution over the vocabulary

Text generation uses nucleus (top-p) sampling with minimum-length constraints.

---

## 3. Architecture Diagram

The following diagram illustrates the overall architecture of the proposed radiology report generation framework, highlighting the integration of visual feature extraction, knowledge graph reasoning, and sequential language generation.

<p align="center">
  <img src="architecture.svg" alt="Multimodal Radiology Report Generation Architecture" width="90%">
</p>

The architecture integrates:
- Dual-view CNN-based visual encoders
- Knowledge graph reasoning via a GCN
- LSTM-based report generation


---

## 4. Repository Structure

```text
.
├── README.md                               # Project overview and documentation
├── .gitignore                             # Git ignore rules
│
├── adjacency_matrix.csv                  # Knowledge graph adjacency matrix
│
├── model_building.ipynb                  # Model training and experimentation notebook
├── knowledge_graph_construction.ipynb    # Knowledge graph construction notebook
├── llm_refinement.ipynb                  # LLM-based report refinement notebook
│
├── results/                              # Generated outputs and evaluation results

```

NOTE: Trained model weights (final_model.pth) are intentionally NOT stored in this repository.  
Please download them from Hugging Face.

---

## 5. Dataset Notice

This repository does NOT include the MIMIC-CXR dataset.

- MIMIC-CXR is subject to PhysioNet credentialed access
- Redistribution of the dataset is prohibited

To obtain the dataset, visit:  
https://physionet.org/content/mimic-cxr/

All training and evaluation code assumes that the user has legally obtained access to the dataset.

---

## 8. Training Code

Training and experimentation code is provided in:

```
model_building.ipynb
```

This notebook includes:
- Knowledge graph construction
- Dataset grouping and pickling
- Data loaders and collate_fn
- Training loop
- Evaluation pipeline
- Model checkpointing

Knowledge graph construction is provided in:

```
knowledge_graph_construction.ipynb
```

Report fine-tuning using LLM is provided in:

```
llm_refinement.ipynb
```

---

## 9. Intended Use

This project is intended for:
- Academic research
- Educational purposes
- Benchmarking multimodal radiology report generation methods

This project is NOT intended for clinical deployment or medical decision-making.

---

## 10. Model Weights

Model weights are available at:

https://huggingface.co/shemzegem/mimic-cxr-report-generator

---
