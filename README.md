# Transformer-By-Hand
A PyTorch-based Transformer implementation designed for learners exploring large language models (LLMs), inspired by [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer).

# Quick Start
Build the repository using Conda. If possible, use Mamba to speed up dependency installation.

**Install mamba**
```
conda install -n base -c conda-forge mamba
```

**Create a virtual environment**
```
conda create -n transformer_env python=3.8  pytorch=1.7.1  cudatoolkit=10.2  torchtext=0.8 -c pytorch
conda activate transformer_env
```

**Install dependencies and vocabularies**
```
mamba install  datasets ipython ipykernel spacy

pip install vocab/de_core_news_sm-3.7.0-py3-none-any.whl 
pip install vocab/en_core_web_sm-3.7.0-py3-none-any.whl 
```

**Run**
```
cd src
python -m Train
```

If everything is set up correctly, you should see output similar to this:
```
025-07-16 08:45:09,201 - INFO - Note: NumExpr detected 64 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
2025-07-16 08:45:09,204 - INFO - NumExpr defaulting to 8 threads.
2025-07-16 08:45:09,594 - INFO - PyTorch version 1.7.1 available.
2025-07-16 08:45:11,288 - INFO - dataset initializing start
Generating train split: 29000 examples [00:00, 1180061.28 examples/s]
Generating validation split: 1014 examples [00:00, 477867.89 examples/s]
Generating test split: 1000 examples [00:00, 606288.52 examples/s]
2025-07-16 08:45:17,658 - INFO - dataset initializing done
2025-07-16 08:54:54,231 - INFO - step : 0.0 %%, loss : 8.681584358215332
2025-07-16 08:54:55,432 - INFO - step : 0.44 %%, loss : 8.681571960449219
2025-07-16 08:54:55,914 - INFO - step : 0.88 %%, loss : 8.681561470031738
2025-07-16 08:54:56,477 - INFO - step : 1.32 %%, loss : 8.681571960449219
...
```


## Directory Structure

```
Transformer-By-Hand/
├── dataset/                        # Datasets for training and evaluation
│   └── multi30k/                   # Multi30k translation dataset (files not listed)
├── result/                         # Output results from training and evaluation
│   ├── bleu.txt                    # BLEU score results for translation quality
│   ├── test_loss.txt               # Test loss values
│   └── train_loss.txt              # Training loss values
├── saved/                          # Saved model checkpoints
│   ├── model-8.357598781585693.pt  # Example model checkpoint (PyTorch format)
├── src/                            # Main source code
│   ├── Train.py                    # Training entry point script
│   ├── Config.py                   # Configuration file for model and training
│   ├── Data.py                     # Data loading and preprocessing
│   ├── util/                       # Utility functions and helpers
│   └── models/                     # Model-related code
│       ├── layers/                 # Basic neural network layers (e.g., attention, feed-forward)
│       ├── blocks/                 # Transformer blocks (encoder/decoder blocks)
│       ├── embedding/              # Embedding and positional encoding layers
│       └── model/                  # High-level model structure
│           ├── Transformer.py      # Transformer architecture (integrates encoder and decoder)
│           ├── Encoder.py          # Encoder implementation
│           └── Decoder.py          # Decoder implementation
├── vocab/                          # Vocabulary and language model files for tokenization
│   ├── en_core_web_sm-3.7.0-py3-none-any.whl
│   ├── de_core_news_sm-3.0.0.tar.gz        # spaCy German model (various versions)
│   ├── de_core_news_sm-3.7.0-py3-none-any.whl
│   ├── de_core_news_sm-3.8.0.tar.gz
```

**Descriptions:**
- **dataset/**: Contains datasets used for training and evaluation, such as Multi30k.
- **result/**: Stores output files from training and evaluation, including loss curves and BLEU scores.
- **saved/**: Holds model checkpoints saved during or after training for later use or resuming.
- **src/**: Main source code for the project, including training scripts, configuration, data processing, and model definitions.
- **vocab/**: Pretrained language models and vocabulary files (mainly for spaCy) used for tokenization and preprocessing.


