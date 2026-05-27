# BERT — Sentiment Analysis (SST-3)

## Project Overview

This project fine-tunes the pretrained `bert-base-uncased` Transformer model for 3-class sentiment classification using the Stanford Sentiment Treebank (SST-3) dataset. The task is to classify movie-review sentences as **negative**, **neutral**, or **positive**.

The primary goal of this project is to compare a modern Transformer-based architecture with a previous hybrid deep learning approach combining CNN, BiLSTM, and Attention mechanisms trained on the same dataset. While the earlier CNN-BiLSTM-Attention model incorporated pretrained GloVe word embeddings, it still learned most task-specific contextual patterns from the SST training data through recurrent sequence modeling. In contrast, BERT starts from deeply pretrained contextual language representations learned from large-scale text corpora and uses self-attention to model relationships between all tokens in a sentence simultaneously.

The fine-tuned BERT model achieved approximately 74% validation accuracy and over 75% test accuracy, improving upon the earlier CNN-BiLSTM-Attention baseline (~65% validation accuracy).

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

The BERT-base model consists of **12 Transformer encoder layers**.
Each encoder layer contains:

- Multi-head self-attention
- Add & Layer Normalization
- Feed-forward neural network
- Add & Layer Normalization

#### Input Processing

Input text is converted into:

- Token embeddings (dimension 768)
- Positional embeddings

These embeddings are passed through the 12 Transformer encoder layers to produce **contextualized token representations**.

For sentiment classification, the final hidden representation of the special **[CLS] token** is used as the sentence-level representation and passed to a classification layer.

#### Architecture Flow

Input Embeddings (768) <br>
→ Positional Embeddings<br>
→ 12 Transformer Encoder Layers<br>
→ Contextualized Token Embeddings<br>
→ [CLS] Token Representation<br>
→ Linear Classification Layer<br>
→ Logits<br>
→ Softmax Probabilities<br>

### Fine-Tuning Pipeline

- Load the SST dataset using torchtext
- Convert sentences into BERT token IDs using the pretrained tokenizer
- Generate padded input sequences and attention masks for Transformer processing
- Initialize BertForSequenceClassification with pretrained BERT-base weights
- Fine-tune the model on SST sentiment classification
- Evaluate performance using validation/test accuracy and confusion matrix analysis

#### Parameters Fine-Tuned During Training

The model starts from pretrained BERT representations learned on large-scale text corpora. During fine-tuning, all Transformer parameters are updated, including:

- Token embeddings
- Positional embeddings
- All 12 Transformer encoder layers
- Query/Key/Value (Q/K/V) projection matrices for every attention head
- Feed-forward network layers
- Layer normalization parameters
- Final classification layer

### Training Configuration

- Optimizer: AdamW
- Learning rate: `2e-5`
- Weight_decay: `0.01`
- Learning‑rate scheduler: Linear warmup (10% of steps) + linear decay
- Batch size: `8`
- Epochs: `2`
- Loss function: CrossEntropyLoss (computed automatically by BERT)

________________________________________

## Comparison between LSTM with CNN and Attention to BERT model

| Model | Architecture | Dataset | Accuracy |
|---|---|---|---|
| CNN-BiLSTM-Attention | Custom hybrid recurrent model | SST-3 | ~65% |
| BERT Fine-Tuning | Pretrained Transformer encoder | SST-3 | ~74–75% |

For the earlier CNN-BiLSTM-Attention implementation, see the [LSTM Sentiment Analysis Repository](https://github.com/elgels/sentiment-analysis-lstm).

### Key Differences

### CNN-BiLSTM-Attention Model
- Used pretrained GloVe word embeddings
- Learned task-specific language patterns primarily from SST training data
- Used recurrent hidden states to model sequential information
- Attention layer highlighted important tokens
- CNN layers extracted local n-gram features from neighboring words

### BERT Transformer Model
- Used pretrained contextual language representations
- Applied bidirectional self-attention across all tokens simultaneously
- Generated contextual sentence representations through the `[CLS]` token
- Captured long-range dependencies more effectively
- Fine-tuned the full pretrained Transformer encoder on SST-3

### Error Analysis

The confusion matrix shows that the model struggled most with the **neutral class**. Many truly negative reviews were predicted as neutral (125 cases), and many truly positive reviews were also predicted as neutral (72 cases). At the same time, truly neutral reviews were frequently classified as positive (116 cases) or negative (151 cases).

In contrast, the model made far fewer direct mistakes between positive and negative sentiment: only 53 negative reviews were classified as positive, and only 26 positive reviews were classified as negative.

This pattern suggests that the model learned to distinguish strong sentiment relatively well, but had difficulty identifying sentences with weak, mixed, or ambiguous emotional signals.

Neutral sentiment is inherently difficult because it is not a strong emotional category like positive or negative. Instead, it often represents ambiguity, subtlety, or the absence of clear sentiment. As a result, both humans and models struggle to classify neutral sentences consistently.

Human agreement studies on the Stanford Sentiment Treebank (SST) also show that annotators agree much less on neutral examples than on clearly positive or negative ones. If humans themselves disagree on what counts as neutral, there is an upper limit on how accurately a model can learn this category.

________________________________________

## Future Improvements

- Fine-tune larger Transformer models such as RoBERTa-base
- Experiment with additional hyperparameter tuning, including learning rate scheduling and batch size optimization
- Visualize attention patterns and token importance across sentences
- Compare performance with additional pretrained Transformer architectures such as DistilBERT

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
├── bert_finetuning_sst3.ipynb
│
├── figures/
│   ├── BERT_Confusion_Matrix.png
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
