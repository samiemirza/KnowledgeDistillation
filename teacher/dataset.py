"""
Dataset and data loading utilities for teacher model activation capture.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import config


class TextDataset(Dataset):
    """
    Dataset for loading and tokenizing text data from a file.
    """

    def __init__(self, file_path, tokenizer, max_length=None):
        """
        Initialize the dataset.

        Args:
            file_path (str): Path to the text file
            tokenizer: Hugging Face tokenizer
            max_length (int): Maximum sequence length. Defaults to config.MAX_SEQ_LENGTH
        """
        self.tokenizer = tokenizer
        self.max_length = max_length or config.MAX_SEQ_LENGTH

        # Read all lines from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.texts)} text samples from {file_path}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a single tokenized sample.

        Args:
            idx (int): Index of the sample

        Returns:
            dict: Tokenized inputs
        """
        text = self.texts[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension (will be added by DataLoader)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def collate_fn(batch):
    """
    Collate function for batching and padding sequences.

    Args:
        batch (list): List of samples from the dataset

    Returns:
        dict: Batched and padded inputs
    """
    # Get max length in this batch for efficient padding
    max_len = max(item['input_ids'].size(0) for item in batch)

    # Prepare lists for batching
    input_ids_list = []
    attention_mask_list = []

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']

        # Calculate padding needed
        padding_length = max_len - input_ids.size(0)

        if padding_length > 0:
            # Pad sequences (padding token is typically 0 or eos_token_id)
            input_ids = torch.cat([
                input_ids,
                torch.zeros(padding_length, dtype=input_ids.dtype)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=attention_mask.dtype)
            ])

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    # Stack into batch tensors
    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list)
    }


def create_dataloader(file_path, tokenizer, batch_size=None, shuffle=False):
    """
    Create a DataLoader for the text dataset.

    Args:
        file_path (str): Path to the text file
        tokenizer: Hugging Face tokenizer
        batch_size (int): Batch size. Defaults to config.BATCH_SIZE
        shuffle (bool): Whether to shuffle the data

    Returns:
        DataLoader: PyTorch DataLoader
    """
    batch_size = batch_size or config.BATCH_SIZE

    dataset = TextDataset(file_path, tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # Use 0 for compatibility with all systems
        pin_memory=torch.cuda.is_available()
    )

    print(f"Created DataLoader with batch_size={batch_size}, {len(dataloader)} batches")

    return dataloader


if __name__ == "__main__":
    # Test dataset (requires a test file)
    print("Dataset module loaded successfully")
    print(f"Default batch size: {config.BATCH_SIZE}")
    print(f"Default max sequence length: {config.MAX_SEQ_LENGTH}")
