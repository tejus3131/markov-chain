# Markov Chain Text Predictor

A sophisticated text prediction system built with Markov chains that provides intelligent text completion and generation capabilities through a web-based interface.

## 🚀 Features

- **Multi-level Tokenization**: Character and word-level tokenization with n-gram support
- **Flexible N-gram Models**: Configurable n-gram sizes (2, 3, 4, 5+ grams) for different prediction contexts
- **Advanced Probability Normalization**: Multiple smoothing techniques (Laplace, Good-Turing)
- **Real-time Web Interface**: Interactive browser-based GUI for testing predictions
- **Text Generation**: Generate coherent text based on seed input
- **Word Completion**: Complete partially typed words intelligently
- **Batch Processing**: Efficient processing of large text datasets
- **Model Persistence**: Save and load trained models

## 🏗️ System Architecture

The system consists of several modular components:

- **Tokenizer**: Multi-level text tokenization (characters, words, n-grams)
- **Knowledge Base**: Stores and manages n-gram frequency statistics
- **Probability Normalizer**: Converts raw counts to normalized probabilities with smoothing
- **Prediction Engine**: Core prediction logic with backoff strategies
- **Web GUI**: Real-time browser interface for testing and interaction

## 📋 Prerequisites

This project requires **uv** for dependency management and execution. If you don't have uv installed, please install it first:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:tejus3131/markov-chain.git
   cd markov
   ```

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

   This will create a virtual environment and install all required dependencies as specified in `pyproject.toml`.

## 🚀 Usage

### Running the Web Interface

Start the interactive web GUI:

```bash
uv run app.py
```

This will:
- Initialize the Markov Chain Text Predictor
- Start a web server (default: http://localhost:8000)
- Automatically open your browser to the interface

### Web Interface Features

The web interface provides:
- **Real-time text prediction** as you type
- **Word and character-level predictions**
- **Text generation** with configurable parameters
- **Word completion** for partial inputs
- **Training data management** - add new text to improve predictions
- **Statistics and analytics** about the model performance

### Programmatic Usage

You can also use the system programmatically:

```python
from chain.chain import MarkovChainTextPredictor

# Initialize the predictor
predictor = MarkovChainTextPredictor(
    ngram_sizes=[2, 3, 4],
    smoothing_method='laplace',
    smoothing_alpha=0.1
)

# Train on sample data
training_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick brown dog jumps over the lazy fox."
]
predictor.train(training_texts)

# Get predictions
result = predictor.predict_next_word("The quick", top_k=5)
print(result)

# Generate text
generated = predictor.generate_text("Hello", max_length=20)
print(generated)
```

## 📁 Project Structure

```
markov/
├── app.py                      # Main launcher script
├── pyproject.toml             # Project configuration and dependencies
├── chain/                     # Core package
│   ├── chain.py              # Main system integration
│   ├── gui.py                # Web interface backend
│   ├── gui.html              # Web interface frontend
│   ├── tokenizer/            # Text tokenization components
│   │   ├── hybrid_tokenizer.py
│   │   ├── character_tokenizer.py
│   │   ├── word_tokenizer.py
│   │   └── types.py
│   ├── knowledge_base/       # N-gram storage and retrieval
│   │   ├── knowledge_base.py
│   │   └── types.py
│   ├── probability_normalizer/ # Probability computation
│   │   ├── normalizer.py
│   │   └── types.py
│   └── prediction_engine/    # Core prediction logic
│       ├── engine.py
│       └── types.py
```

## ⚙️ Configuration

The system supports various configuration options:

- **N-gram sizes**: Configure which n-gram sizes to use (default: [2, 3, 5, 7])
- **Smoothing method**: Choose between 'laplace', 'good_turing', or 'none'
- **Smoothing alpha**: Smoothing parameter for Laplace smoothing (default: 0.1)
- **Server port**: Web interface port (default: 8000)

## 🧪 Example Use Cases

- **Autocomplete systems**: Enhance text editors with intelligent completion
- **Writing assistance**: Help with creative writing and content generation
- **Language modeling**: Research and experimentation with n-gram models
- **Text analysis**: Analyze writing patterns and predict next words

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Built with Python 3.13+ and managed with [uv](https://github.com/astral-sh/uv).