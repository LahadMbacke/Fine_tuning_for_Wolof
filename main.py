
# pip install transformers datasets torch peft bitsandbytes accelerate

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

class WolofTranslatorTrainer:
    def __init__(
        self,
        base_model="Undi95/Meta-Llama-3.1-8B-Claude",
        dataset_name="galsenai/french-wolof-translation",
        output_dir="wolof_translator"
    ):
        self.base_model = base_model
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        
    def load_and_prepare_data(self):
        print("Chargement du dataset GalsenAI...")
        self.dataset = load_dataset(self.dataset_name)
        print(f"Dataset chargé. Nombre d'exemples: {len(self.dataset['train'])}")
        
        # First, let's inspect the actual structure of the dataset
        print("\nStructure du dataset:")
        print(f"Colonnes disponibles: {self.dataset['train'].column_names}")
        
        # Display a few examples with proper error handling
        print("\nExemples du dataset:")
        try:
            for i, example in enumerate(self.dataset['train'][:3]):
                print(f"\nExemple {i+1}:")
                # Access the correct column names based on dataset structure
                for key in example.keys():
                    print(f"{key}: {example[key]}")
        except Exception as e:
            print(f"Erreur lors de l'affichage des exemples: {str(e)}")
            print("Structure d'un exemple:", dict(self.dataset['train'][0]))
    
    def prepare_model(self):
        print("\nPréparation du modèle...")
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with 8-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Prepare for fine-tuning
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        
    def preprocess_data(self):
        print("Préparation des données d'entraînement...")
        
        def format_translation(example):
            # Get the correct column names from the dataset
            source_text = example.get('text_fr', example.get('french', ''))
            target_text = example.get('text_wo', example.get('wolof', ''))
            
            # Format: "Traduire du français vers le wolof : {french} => {wolof}"
            return {
                "text": f"Traduire du français vers le wolof : {source_text} => {target_text}"
            }
        
        # Apply formatting
        self.processed_dataset = self.dataset.map(
            format_translation,
            remove_columns=self.dataset["train"].column_names
        )
        
        # Tokenization
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt"
            )
        
        self.tokenized_dataset = self.processed_dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"]
        )
    
    def train(self):
        print("Configuration de l'entraînement...")
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=100,
            save_steps=500,
            warmup_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"] if "validation" in self.tokenized_dataset else None,
            data_collator=data_collator,
        )
        
        print("Début de l'entraînement...")
        trainer.train()
        
        # Save the model
        print(f"Sauvegarde du modèle dans {self.output_dir}")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
    def run_training_pipeline(self):
        """Execute the complete training pipeline"""
        self.load_and_prepare_data()
        self.prepare_model()
        self.preprocess_data()
        self.train()

def main():
    trainer = WolofTranslatorTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()