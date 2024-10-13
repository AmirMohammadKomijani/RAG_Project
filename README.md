# NLP for FAQ Chatbot with Advanced Language Models

## Project Overview
This project focuses on building a highly accurate and efficient FAQ chatbot using state-of-the-art language models, trained and tested on the **SQuAD v2** dataset. The chatbot is designed to understand user queries and provide relevant responses by leveraging both retrieval-based and generative NLP techniques. This project demonstrates expertise in model handling, prompt engineering, and retrieval-based NLP, which are critical skills for an NLP specialist.

### Key Objectives
- Utilize large language models for natural language understanding and generation.
- Implement a retrieval-based system to enhance response relevance and accuracy.
- Optimize for GPU memory usage and performance through mixed precision and model offloading techniques.

## Models Used
The chatbot integrates multiple pre-trained large language models (LLMs) for generating responses:
1. **GPT-2 (1.5B Parameters)**: A baseline model for general natural language processing tasks.
2. **Qwen/Qwen2.5-Coder-7B-Instruct**: A high-performance model for instruction-following tasks, suitable for structured FAQ responses.
3. **Microsoft Phi-3.5-Mini-Instruct**: A smaller, efficient model for faster inference.

These models were selected for their ability to generalize across diverse queries and contexts, offering a balanced trade-off between performance and computational cost.

## Data: SQuAD v2
The **Stanford Question Answering Dataset (SQuAD) v2** was used to train and validate the chatbot. SQuAD v2 is a widely used benchmark dataset containing over 100,000 question-answer pairs based on a set of Wikipedia articles. It includes questions that may or may not have answers, challenging the model to handle unanswerable questions gracefully.

## Architecture and Components
### 1. **Response Generators**
   The models were integrated as response generators using the Hugging Face `transformers` pipeline. Fine-tuning and prompt engineering were applied to maximize the coherence and relevance of generated responses.

### 2. **Retriever**
   A retriever was developed to fetch contextually similar entries from SQuAD v2, enhancing the chatbot's ability to provide accurate responses. The retriever employs cosine similarity with sentence embeddings to find the most relevant content from the dataset.

### 3. **Memory Optimization**
   To handle large models on GPU, the project leveraged:
   - **Mixed Precision (bfloat16)**: This reduces the memory footprint while maintaining numerical stability.
   - **Model Offloading and Device Mapping**: Efficiently distributing model components across available GPU and CPU resources.

### 4. **Pipeline**
   The pipeline follows these steps:
   - **Context Retrieval**: Finds relevant passages from SQuAD v2 to enhance understanding.
   - **Response Generation**: Produces detailed answers based on the retrieved context and user query.

## Installation and Setup
To set up the environment, install the required libraries:
```bash
pip install transformers datasets sentence-transformers
