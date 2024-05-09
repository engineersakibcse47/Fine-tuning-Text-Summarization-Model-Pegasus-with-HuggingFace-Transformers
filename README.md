# Fine-tuning Text Summarization Model- Pegasus with HuggingFace Transformers

### Introduction

This project aimed to fine-tune the Pegasus model on the SAMSum (Self-Annotated Multi-Modal Summarization) dataset, a collection of dialogues paired with summaries. The objective was to adapt the pre-trained Pegasus model to generate accurate and concise summaries for dialogues.

### Methodology

1. Dependency Installation
We installed necessary dependencies using pip, including the Transformers library for model handling, datasets for dataset loading, and evaluation metrics such as ROUGE.

2. Model Selection and Loading
The Pegasus model (google/pegasus-cnn_dailymail) was selected for its effectiveness in abstractive summarization tasks. We loaded the pre-trained model along with its tokenizer.

3. Dataset Preparation
The SAMSum dataset was loaded using the datasets library. It consists of train, test, and validation splits, each containing dialogues and their corresponding summaries.

4. Preprocessing and Tokenization
We prepared the dataset for training by defining a preprocessing function to tokenize dialogues and summaries. This function encoded the data into input-output pairs suitable for sequence-to-sequence tasks.

5. Training
Training was performed using the Trainer module from the Transformers library. We configured training parameters such as batch size, number of epochs, and evaluation strategy. The model was trained on the training split of the SAMSum dataset.

6. Evaluation
The fine-tuned model was evaluated using the ROUGE metric on the test set. ROUGE scores were calculated to assess the quality of the generated summaries compared to the ground truth.

### Conclusion
In this project, we successfully fine-tuned the Pegasus model on the SAMSum dataset. However, the obtained ROUGE scores indicate that the model's performance may not meet the desired standards. Further analysis and experimentation may be needed to improve the summarization quality, potentially through adjustments in training strategies, hyperparameters tuning, or dataset augmentation.


