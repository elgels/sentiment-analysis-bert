# BERT — Sentiment Analysis (SST-3)

## Project Overview

This project fine-tunes the pretrained `bert-base-uncased` Transformer model for 3-class sentiment classification using the Stanford Sentiment Treebank (SST-3) dataset. The task is to classify movie-review sentences as **negative**, **neutral**, or **positive**.

The primary goal of this project is to compare a modern Transformer-based architecture with a previous hybrid deep learning approach combining CNN, BiLSTM, and Attention mechanisms trained on the same dataset. While the earlier CNN-BiLSTM-Attention model incorporated pretrained GloVe word embeddings, it still learned most task-specific contextual patterns from the SST training data through recurrent sequence modeling. In contrast, BERT starts from deeply pretrained contextual language representations learned from large-scale text corpora and uses self-attention to model relationships between all tokens in a sentence simultaneously.

The fine-tuned BERT model achieved approximately 71% validation accuracy and over 72% test accuracy, improving upon the earlier CNN-BiLSTM-Attention baseline (~64% validation accuracy).

________________________________________

## Dataset

The project uses the **Stanford Sentiment Treebank (SST-3)** dataset through `torchtext.datasets.SST`.

### Sentiment Classes

| Label | Description |
|---|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

The SST-3 task is more challenging than binary sentiment classification because many neutral sentences contain subtle or ambiguous sentiment patterns.

________________________________________

## Model Architecture

This project fine-tunes the pretrained `bert-base-uncased` model from Hugging Face Transformers.

### BERT Architecture

- 12 Transformer encoder layers
- Hidden size: 768
- Multi-head self-attention
- Pretrained contextual token embeddings
- `[CLS]` token used for sentence-level classification

### Fine-Tuning Pipeline

1. Load SST dataset using `torchtext`
2. Convert sentences into BERT token IDs using the pretrained tokenizer
3. Create attention masks and padded sequences
4. Fine-tune BERT using `BertForSequenceClassification`
5. Evaluate performance using validation/test accuracy and confusion matrix analysis

### Training Configuration

- Optimizer: AdamW
- Learning rate: `2e-5`
- Weight_decay: `0.01`
- Batch size: `8`
- Epochs: `2`
- Loss function: CrossEntropyLoss (computed automatically by BERT)

________________________________________

## Comparison between LSTM with CNN and Attention to BERT model

| Model | Architecture | Dataset | Accuracy |
|---|---|---|---|
| CNN-BiLSTM-Attention | Custom hybrid recurrent model | SST-3 | ~64% |
| BERT Fine-Tuning | Pretrained Transformer encoder | SST-3 | ~71–75% |

### Key Differences

### CNN-BiLSTM-Attention Model
- Used pretrained GloVe word embeddings
- Learned task-specific language patterns primarily from SST training data
- Used recurrent hidden states to model sequential information
- Attention layer highlighted important tokens
- CNN layers extracted local phrase-level features

### BERT Transformer Model
- Used pretrained contextual language representations
- Applied bidirectional self-attention across all tokens simultaneously
- Generated contextual sentence representations through the `[CLS]` token
- Captured long-range dependencies more effectively
- Fine-tuned the full pretrained Transformer encoder on SST-3

### Error Analysis

The main source of classification error involved the **neutral class**. The model frequently confused neutral examples with mildly positive or mildly negative sentiment, while direct confusion between strongly positive and strongly negative examples was much lower.

This suggests that the model successfully learned overall sentiment polarity but struggled with ambiguous middle-category sentiment, which is a known challenge in SST-3 classification tasks.

________________________________________

## Future Improvements

- Fine-tune larger Transformer models such as RoBERTa-base
- Experiment with learning rate scheduling and batch size tuning
- Add precision, recall, and F1-score analysis
- Perform hyperparameter optimization
- Visualize attention patterns and token importance
- Compare additional pretrained Transformer architectures such as DistilBERT

________________________________________

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- torchtext
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- tqdm
- Jupyter Notebook

________________________________________

## Repository Structure

```text
sentiment-analysis-bert/
│
├── notebooks/
│   └── bert_finetuning_sst3.ipynb
│
├── figures/
│   ├── confusion_matrix.png
│   └── training_results.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

________________________________________

## Key Takeaways

- Fine-tuning pretrained Transformer models can significantly improve NLP performance compared to recurrent architectures trained from scratch.
- BERT’s self-attention mechanism provides stronger contextual language representations than sequential hidden-state modeling alone.
- The SST-3 dataset remains challenging due to ambiguity within the neutral sentiment class.
- Confusion matrix analysis showed that most remaining classification errors involved neutral sentiment rather than direct positive/negative confusion.

________________________________________

## License

This project is licensed under the MIT License.
