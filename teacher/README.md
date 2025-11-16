# Teacher Model - Knowledge Distillation

This module handles the teacher model (DeepSeek-R1-Distill-Llama-8B) for knowledge distillation. It captures layer activations during forward passes and saves them for training the student model.

## Directory Structure

```
teacher/
├── __init__.py              # Module initialization
├── config.py                # Configuration settings
├── teacher_model.py         # Model loading utilities
├── hooks.py                 # Activation capture hooks
├── dataset.py               # Dataset and data loading
├── dump_activations.py      # Main script for dumping activations
├── activations/             # Saved activations (created automatically)
└── offload/                 # Model offloading directory (if enabled)
```

## Features

- **Automatic Device Detection**: Automatically uses CUDA if available, falls back to CPU
- **Memory Efficient**: Supports disk offloading for large models
- **Flexible Layer Selection**: Configure which layers to capture activations from
- **Batch Processing**: Processes data in batches and saves incrementally
- **Progress Tracking**: Uses tqdm for visual progress indication

## Configuration

Edit `config.py` to customize:

```python
# Model settings
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Data settings
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 512

# Which layers to capture (adjust based on your needs)
LAYERS_TO_HOOK = [0, 8, 16, 24, 31]

# Output directory
ACTIVATION_DIR = "teacher/activations"

# Memory management
USE_DISK_OFFLOAD = False  # Set to True if running out of memory
```

## Usage

### Basic Usage

```bash
cd teacher
python dump_activations.py --data_file /path/to/your/data.txt
```

### Advanced Usage

```bash
python dump_activations.py \
    --data_file /path/to/data.txt \
    --output_dir ./my_activations \
    --num_batches 100 \
    --batch_size 8 \
    --max_seq_length 1024
```

### Arguments

- `--data_file` (required): Path to input text file (one sample per line)
- `--output_dir`: Directory to save activations (default: `teacher/activations`)
- `--num_batches`: Limit number of batches to process (default: process all)
- `--batch_size`: Batch size (default: from config.py)
- `--max_seq_length`: Maximum sequence length (default: from config.py)

## Data Format

The input text file should contain one training sample per line:

```
This is the first training sample.
This is the second training sample.
Each line will be tokenized separately.
```

## Output Format

Activations are saved as PyTorch tensors in `.pt` files:

```
activations/
├── batch_000000.pt
├── batch_000001.pt
├── batch_000002.pt
└── ...
```

Each file contains a dictionary with layer activations:

```python
{
    'layer_0': torch.Tensor,   # Shape: [batch_size, seq_length, hidden_size]
    'layer_8': torch.Tensor,
    'layer_16': torch.Tensor,
    ...
}
```

## Loading Saved Activations

```python
import torch

# Load activations for a specific batch
activations = torch.load('teacher/activations/batch_000000.pt')

# Access specific layer
layer_0_acts = activations['layer_0']
print(layer_0_acts.shape)  # [batch_size, seq_length, hidden_size]
```

## Memory Management

### GPU Memory

If running out of GPU memory, try:

1. Reduce batch size in `config.py`
2. Reduce sequence length
3. Enable disk offloading: `USE_DISK_OFFLOAD = True`

### CPU Memory

For CPU-only systems:

1. Use smaller batch sizes
2. Process data in chunks using `--num_batches`
3. Clear activations directory between runs if space is limited

## Testing Individual Components

### Test Model Loading

```bash
python teacher_model.py
```

### Test Hook System

```bash
python hooks.py
```

### Test Dataset

```bash
python dataset.py
```

## Requirements

- PyTorch >= 2.0
- transformers >= 4.30
- tqdm
- CUDA (optional, for GPU acceleration)

Install with:

```bash
pip install torch transformers tqdm
```

## Troubleshooting

### Model Download Issues

If the model fails to download:
- Check your internet connection
- Ensure you have Hugging Face access (may require login for some models)
- Try: `huggingface-cli login`

### Out of Memory

- Reduce `BATCH_SIZE` in config.py
- Enable `USE_DISK_OFFLOAD = True`
- Use `--num_batches` to process in smaller chunks
- Reduce `MAX_SEQ_LENGTH`

### Slow Processing

- Use GPU if available (check with `torch.cuda.is_available()`)
- Increase batch size if memory allows
- Consider reducing number of layers to hook

## Next Steps

After dumping activations, use them to train the student model with the distillation scripts in the `distill/` directory.
