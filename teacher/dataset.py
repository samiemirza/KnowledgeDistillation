"""
Dataset and data loading utilities for teacher model activation capture.
Incorporates Official Paper Formatting (DeepSeek Control Tokens & System Prompts).
"""

import torch
import json
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import config

# ==============================================================================
#  Formatting Logic (Stolen from Paper's Repo)
# ==============================================================================

class PaperFormatter:
    """
    Handles the specific prompt engineering used in the paper 
    to trigger DeepSeek-R1's reasoning capabilities.
    """
    
    # The official template DeepSeek expects
    CHAT_TEMPLATE = """{system}<｜User｜>{user}<｜Assistant｜><think>
{assistant_reasoning}
</think>
{assistant_solution}"""

    # The magic spell that makes the model think hard
    SYSTEM_PROMPT = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process
before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of
analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered
thinking process.
"""

    def format_openthoughts(self, sample):
        """
        Formats a sample from OpenThoughts using the specific domain logic.
        """
        domain = sample.get('domain', 'math') # Default to math if missing
        
        # 1. Format the User Input based on domain
        user_input = ""
        
        if domain == 'code':
            # Code specific prompt
            try:
                test_cases = json.loads(sample.get("test_cases", "{}"))
                if not test_cases.get("fn_name"):
                    user_input += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."
                else:
                    user_input += "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
            except:
                pass
            
            user_input += sample.get("problem", "")
            if sample.get("starter_code"):
                user_input += "\n" + sample["starter_code"]
                
        elif domain == 'math':
            # Math specific prompt
            user_input = f"Return your final response within \\boxed{{}}. {sample.get('problem', '')}"
            
        else:
            # Puzzle/Science/General
            user_input = sample.get('problem', '')

        # 2. Apply the Full Template with <think> tags
        return self.CHAT_TEMPLATE.format(
            system=self.SYSTEM_PROMPT,
            user=user_input,
            assistant_reasoning=sample.get("deepseek_reasoning", ""),
            assistant_solution=sample.get("deepseek_solution", "")
        )

    def format_lmsys(self, sample):
        """
        Formats LMSYS chat using DeepSeek special tokens.
        """
        conversation = sample.get("conversation", [])
        prompt = []
        
        # Their script logic for LMSYS
        for message in conversation:
            role = message['role']
            content = message['content']
            
            if role == 'system':
                prompt.append(content)
            elif role == 'user':
                prompt.append('<｜User｜>' + content)
            elif role == 'assistant':
                prompt.append('<｜Assistant｜>' + content)
                # Note: They add <|end of sentence|> in the repo script
                # We let the tokenizer handle EOS, or add it manually if needed.
                # Usually standard chat templates handle this, but explicit is fine.
                prompt.append('<｜end of sentence｜>')

        return "".join(prompt)

# ==============================================================================
#  Dataset Class
# ==============================================================================

class HFActivationDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, max_length=None, split="train", max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length or config.MAX_SEQ_LENGTH
        self.dataset_name = dataset_name
        self.formatter = PaperFormatter()
        
        print(f"Loading dataset: {dataset_name}...")
        
        if dataset_name == "openthoughts":
            # Load metadata to get reasoning columns
            self.data = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split=split)
            # Filter domains like the paper did (optional, but good for consistency)
            valid_domains = ['code', 'math', 'puzzle', 'biology', 'chemistry', 'physics']
            self.data = self.data.filter(lambda x: x.get('domain') in valid_domains)
            
        elif dataset_name == "lmsys":
            self.data = load_dataset("lmsys/lmsys-chat-1m", split=split)
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if max_samples:
            self.data = self.data.select(range(min(len(self.data), max_samples)))
            
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 1. Format Text using Paper Logic
        if self.dataset_name == "openthoughts":
            text = self.formatter.format_openthoughts(sample)
        else:
            text = self.formatter.format_lmsys(sample)

        # 2. Tokenize
        # Note: trust_remote_code=True in model loader ensures tokenizer 
        # understands <|User|> etc.
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def create_dataloader(dataset_name, tokenizer, batch_size=None, max_samples=None):
    batch_size = batch_size or config.BATCH_SIZE
    dataset = HFActivationDataset(dataset_name, tokenizer, max_samples=max_samples)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
