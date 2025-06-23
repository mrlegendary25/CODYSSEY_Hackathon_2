import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import argparse
import sys
import json
from datasets import load_dataset
import google.generativeai as genai
import logging

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the custom model architecture
class BERTModel(nn.Module):
    def __init__(self, num_labels):
        super(BERTModel, self).__init__()
        self.transformer = AutoModel.from_pretrained("roberta-base")
        self.fc1 = nn.Linear(self.transformer.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return self.sigmoid(logits)

# Load dataset to get emotion labels
dataset = load_dataset("go_emotions", split="test")
emotion_labels = dataset.features["labels"].feature.names

# Define Ekman Mapping
ekman_mapping = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "gratitude", "love", "optimism", "relief", "pride"],
    "sadness": ["sadness", "disappointment", "grief"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"]
}

def map_to_ekman(goemotions_probs):
    """Map GoEmotions probabilities to Ekman emotions"""
    ekman_probs = {emotion: 0 for emotion in ekman_mapping.keys()}
    for ekman_emotion, goemotions_list in ekman_mapping.items():
        ekman_probs[ekman_emotion] = sum(
            [goemotions_probs.get(label, 0) for label in goemotions_list]
        )
    return ekman_probs

def classify_emotion(text):
    """Classify emotions in the given text"""
    # Load tokenizer and model
    #tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer")
    #model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    tokenizer_path = "C:\\Users\\Prashant\\Documents\\Himnish\\Web Development\\C-ODYSSEY\\tokenizer"
    model_path="C:\\Users\\Prashant\\Documents\\Himnish\\Web Development\\C-ODYSSEY\\model.pth"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = BERTModel(num_labels=len(emotion_labels))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        raise Exception(f"Error loading model or tokenizer: {str(e)}")

    try:
        # Tokenize and prepare input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Convert to probabilities
        probabilities = outputs.cpu().numpy()[0]
        
        # Create emotion probabilities dictionary
        goemotions_probs = {label: float(prob) for label, prob in zip(emotion_labels, probabilities)}
        
        # Get Ekman emotions
        ekman_probs = map_to_ekman(goemotions_probs)
        
        # Get top 3 emotions
        top_3_emotions = sorted(goemotions_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return goemotions_probs, ekman_probs, top_3_emotions
    
    except Exception as e:
        raise Exception(f"Error during emotion classification: {str(e)}")

def main():
    try:
        #print('came here2')
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--text', type=str, required=True, help='Text to analyze or path to file containing text')

        args = parser.parse_args()
        #print('here 3')
        # Read text from file if it exists, otherwise use direct text
        try:
            with open(args.text, 'r', encoding='utf-8') as f:
                #print('came here also to read text')
                text = f.read()

        except (IOError, OSError) as e:
            raise Exception(f"Failed to read input file: {e}")
        
        if not text.strip():
            raise ValueError("Empty text provided")
        
        if 10==20:
            print("10 is less than 20")

        
        # Perform emotion analysis
        goemotions_probs, ekman_probs, top_3_emotions = classify_emotion(text)
        
        # Prepare results
        result = {
            'goemotions': {str(k): float(v) for k, v in goemotions_probs.items()},
            'ekman': {str(k): float(v) for k, v in ekman_probs.items()},
            'top_3': [(str(e), float(p)) for e, p in top_3_emotions]
        }
        
        # Output JSON results
        print(json.dumps(result))
        return 0
        
    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())