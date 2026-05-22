import os
from pathlib import Path
import pandas as pd
import json
from dotenv import load_dotenv

db = pd.read_csv("./cyberbullying_tweets.csv")
db.head()

from groq import Groq
dotenv_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=dotenv_path, override=True)
secret_key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
if not secret_key:
    raise ValueError("GROQ_API_KEY not found. Add it to ArtificialIntelligence/HW3/.env")

# user prompt
system_imput = """
You are an expert in identifying cyberbullying. You will get a list of tweets as input and your task is to identify if ot was cyberbullying or not. If if it classified as cyber bullying then identify which type of cyber bullying is it, 
your output should be strictly a valid JSON, do not add anything else example: {ethnicity} where you classify the cyberbullying as one of these labels: 
not_cyberbullying
gender
religion
other_cyberbullying
age
ethnicity
"""
tweets = db["tweet_text"]
all_prompts = []
# select 10000 values from the dataset and create prompts
for chunk in tweets[0:10]:
    chunk_str = str(chunk) 
    
    prompt_content = [
        {"role": "system", "content": system_imput},
        {"role": "user", "content": f"Tweet: {chunk_str}"}
    ]
    all_prompts.append(prompt_content)


client = Groq(api_key=secret_key)

# call LLM
def process_with_groq(messages):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=messages, 
            temperature=0, 
            max_tokens=100,
            response_format={"type": "json_object"} 
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return None

final_outputs = []
for p in all_prompts:
    result = process_with_groq(p)
    final_outputs.append(result)

print("All results received!")

y_pred_zero_shot = [item.split(': "')[1].split('"')[0] if isinstance(item, str) and ': "' in item else "error" for item in final_outputs]

# User input with few examples
system_imput = """
You are an expert in identifying cyberbullying. You will get a list of tweets as input and your task is to identify if ot was cyberbullying or not. If if it classified as cyber bullying then identify which type of cyber bullying is it, 
your output should be strictly a valid JSON, do not add anything else example: {ethnicity} where you classify the cyberbullying as one of these labels: not_cyberbullying, gender, religion, other_cyberbullying, age, ethnicity
Input:
Tweet: Need what he's smoking @RajAshok5 Being feminist isnt sexist BUT ASKING LAWS &amp; INSISTING THAT WOMEN ARE CORRECT ALWAYS, MEN ARE CRIMINALS IS	
Output:
gender
Input:
@Raja5aab @Quickieleaks Yes, the test of god is that good or bad or indifferent or weird or whatever, it all proves gods existence.
Output:
not_cyberbullying
Input:
In Islam women must be locked in their houses, and Muslims claim this is treating them well.
Output:
religion
Input:
@LeoKikiLady89 Funniest commercial I ever saw...until they did the Maury one.
Output:
not_cyberbullying
Input:
Girl who bullied me in high school just asked me for hair bleaching tips. Imma tell her 40 volume and to start at her roots
Output:
age
"""
tweets = db["tweet_text"]
all_prompts = []
# select 10000 values from the dataset and create prompts
# for chunk in tweets[0:10000]:
for chunk in tweets[0:10]:
    chunk_str = str(chunk) 
    
    prompt_content = [
        {"role": "system", "content": system_imput},
        {"role": "user", "content": f"Tweet: {chunk_str}"}
    ]
    all_prompts.append(prompt_content)


client = Groq(api_key=secret_key)

# calling LLM
def process_with_groq(messages):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=messages, 
            temperature=0, 
            max_tokens=100,
            response_format={"type": "json_object"} 
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return None

final_outputs = []
for p in all_prompts:
    result = process_with_groq(p)
    final_outputs.append(result)

print("All results received!")

y_pred_few_shot = [item.split(': "')[1].split('"')[0] if isinstance(item, str) and ': "' in item else "error" for item in final_outputs]

import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

# 1. Get the true labels 
y_true = db['cyberbullying_type'][0:10].tolist()

# 2. Clean up the model outputs 
y_pred_zero_shot = [str(item).strip().lower() for item in y_pred_zero_shot]
y_pred_few_shot = [str(item).strip().lower() for item in y_pred_few_shot]
y_true_clean = [str(item).strip().lower() for item in y_true]

# 4. Create a side-by-side comparison DataFrame 
precision_zs, recall_zs, f1_zs, _ = precision_recall_fscore_support(y_true_clean, y_pred_zero_shot, average='weighted', zero_division=0)
precision_fs, recall_fs, f1_fs, _ = precision_recall_fscore_support(y_true_clean, y_pred_few_shot, average='weighted', zero_division=0)

comparison_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score'],
    'Zero-Shot (Weighted)': [precision_zs, recall_zs, f1_zs],
    'Few-Shot (Weighted)': [precision_fs, recall_fs, f1_fs]
})

print("\n================ SUMMARY COMPARISON ================")
print(comparison_df)