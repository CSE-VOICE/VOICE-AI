import pandas as pd
import torch 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import os
from peft import get_peft_model, LoraConfig, TaskType

# Cuda GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Apple Silicon GPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device : {device}")


# Load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    return df

def print_trainable_parameters(model, unit=1e9):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params/unit:.1f}B || all params: {all_param/unit:.1f}B || trainable%: {100 * trainable_params / all_param}"
    )

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        situation = self.data.iloc[index]['situation']
        routine = self.data.iloc[index]['routine']

        inputs = self.tokenizer(situation, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        labels = self.tokenizer(routine, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')

        return {
            'input_ids': inputs.input_ids[0],
            'labels': labels.input_ids[0]
        }

DATASET_PATH = "../preprocessed_dataset.csv"

# Load dataset
df = load_data(DATASET_PATH)

# Load Tokenizer & Model
tokenizer = T5TokenizerFast.from_pretrained('paust/pko-chat-t5-large')
model = T5ForConditionalGeneration.from_pretrained('paust/pko-chat-t5-large', torch_dtype=torch.bfloat16)
model.to(device)

# Config LoRA
lora_config = LoraConfig(
    task_type = TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q", "v", "k", "o"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()



# Hyperparameters
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 10
log_interval = 100
batch_size = 4

# split train & validation dataset
train_data, val_data = train_test_split(df, test_size=0.05, random_state=42)

# Load data using DataLoader
train_dataset = CustomDataset(train_data, tokenizer)
val_dataset = CustomDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Create new directory for saving model checkpoints if not exist
checkpoint_dir = "model_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# model train
total_steps = 0
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader, total=len(train_loader)):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

       
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Train using GradScaler
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_steps += 1

        if total_steps % log_interval == 0:
            print(f"Epoch {epoch + 1}, Step {total_steps}, Training Loss: {loss.item()}")

        # save checkpoint for every 0.5 epoch
        if total_steps % int(len(train_dataset) / (batch_size * 2)) == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{total_steps}")
            model.save_pretrained(checkpoint_path)


    # validation part
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            val_loss += outputs.loss.item() * batch['input_ids'].size(0)

    val_loss /= len(val_dataset)
    print(f"Epoch{epoch + 1} : Validation Loss : {val_loss}")


model.save_pretrained(os.path.join(checkpoint_dir, "final_model"))