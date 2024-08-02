
## Text Summarization - Fine Tuning Google Pegasus Model on Custom Dataset to Higgingface Transformer. 

## Introduction

The aim of this project is to fine-tune the Google Pegasus model, a state-of-the-art summarization model, on a custom dataset. This involves several steps, including dependency installation, model downloading, dataset loading, preprocessing, and training. Finally, the model's performance will be evaluated using the ROUGE score.

## Dependency Installation

To begin with, the necessary dependencies must be installed. This typically includes libraries such as `transformers` from Hugging Face, which provides the Pegasus model and related tools for text processing and training.

```bash
pip install transformers
pip install datasets
pip install rouge_score
```

## Download Model

The Pegasus model is then downloaded from the Hugging Face model repository. This pre-trained model serves as the base for further fine-tuning on the specific dataset.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
```

## Load Dataset

The custom dataset is loaded for training the model. This dataset should be in a format suitable for text summarization tasks, with input texts and corresponding summaries.

```python
from datasets import load_dataset

samsum = load_dataset('samsum')
```

## Preprocess Dataset

### Visualize Length

Before preprocessing, it is helpful to visualize the length of the texts and summaries in the dataset. This can guide decisions about truncation or padding during preprocessing.

```python
dialogue_len = [len(x['dialogue'].split()) for x in samsum['train']]
summary_len = [len(x['summary'].split()) for x in samsum['train']]
import pandas as pd

data = pd.DataFrame([dialogue_len, summary_len]).T
data.columns = ['Dialogue Length', 'Summary Length']

data.hist(figsize=(15,5))
```

### Tokenization

A `preprocess_function()` is defined to tokenize the entire corpus. The `Dataset.map()` function from the `datasets` library is used to apply this preprocessing to the dataset efficiently.

```python
def preprocess_function(example_batch):

    encodings = tokenizer(example_batch['dialogue'], text_target=example_batch['summary'],
                        max_length=1024, truncation=True)

    encodings = {'input_ids': encodings['input_ids'],
               'attention_mask': encodings['attention_mask'],
               'labels': encodings['labels']}

    return encodings
```

## Build Trainer

A trainer is built using the `Trainer` class from the `transformers` library. This class facilitates the training process by managing the model, optimizer, and training loop. Key parameters, such as learning rate and batch size, are set during this step.

```python
trainer = Trainer(model=model_pegasus,
                  args=training_args,
                  tokenizer=tokenizer,
                  data_collator=data_collator,
                  train_dataset = samsum_tokenize_pt['train'],
                  eval_dataset = samsum_tokenize_pt['validation'])
```

## Test Model Pipeline

After training, the model is tested on a validation set to assess its performance. This involves generating summaries for the validation texts and comparing them to the reference summaries.

```python
from transformers import pipeline
gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}
pipe = pipeline("summarization", model="pegasus-samsum-model",tokenizer=tokenizer)
```

## Evaluate Performance

### Calculate ROUGE Score

The ROUGE score, a common metric for evaluating summarization models, is calculated to quantify the model's performance. The ROUGE score compares the overlap of n-grams between the generated summaries and the reference summaries, providing insight into the model's accuracy and coherence.

```python
rouge_metric = load_metric('rouge')

score = calculate_metric_on_test_ds(
    samsum['test'], rouge_metric, trainer.model, tokenizer, batch_size = 2, column_text = 'dialogue', column_summary= 'summary')

rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )
pd.DataFrame(rouge_dict, index = [f'pegasus'] )
```

## Conclusion
The project successfully fine-tuned the Pegasus model on a custom dataset for summarization. Key steps included dependency installation, dataset preprocessing, and model training using the transformers library. The model's performance was evaluated using ROUGE scores. Future work could explore further hyperparameter tuning and dataset augmentation to enhance model accuracy.
