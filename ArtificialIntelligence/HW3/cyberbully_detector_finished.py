import os
from pathlib import Path
import pandas as pd
import json
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

db = pd.read_csv("./cyberbullying_tweets.csv")

LABELS = [
    "not_cyberbullying",
    "gender",
    "religion",
    "other_cyberbullying",
    "age",
    "ethnicity",
]

SAMPLE_SIZE = 100
SUPPORT_START = 100
SUPPORT_SIZE = 900
MAX_TOKENS = 100
MODEL_NAME = "llama-3.3-70b-versatile"

dotenv_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=dotenv_path, override=True)
secret_key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
if not secret_key:
    raise ValueError("GROQ_API_KEY not found. Add it to ArtificialIntelligence/HW3/.env")

client = Groq(api_key=secret_key)

SYSTEM_PROMPT = """
You are an expert in identifying cyberbullying in tweets.
Classify each tweet into exactly one label from:
not_cyberbullying, gender, religion, other_cyberbullying, age, ethnicity.
Return ONLY valid JSON in the format: {"label": "one_label_here"}
""".strip()

STATIC_FEW_SHOT_EXAMPLES = [
    (
        "Need what he's smoking @RajAshok5 Being feminist isnt sexist BUT ASKING LAWS &amp; "
        "INSISTING THAT WOMEN ARE CORRECT ALWAYS, MEN ARE CRIMINALS IS",
        "gender",
    ),
    (
        "@Raja5aab @Quickieleaks Yes, the test of god is that good or bad or indifferent or weird "
        "or whatever, it all proves gods existence.",
        "not_cyberbullying",
    ),
    (
        "In Islam women must be locked in their houses, and Muslims claim this is treating them well.",
        "religion",
    ),
    (
        "@LeoKikiLady89 Funniest commercial I ever saw...until they did the Maury one.",
        "not_cyberbullying",
    ),
    (
        "Girl who bullied me in high school just asked me for hair bleaching tips. "
        "Imma tell her 40 volume and to start at her roots",
        "age",
    ),
]


def extract_label(response_text):
    if not isinstance(response_text, str):
        return "error"
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return "error"
    if not isinstance(payload, dict) or not payload:
        return "error"
    for key in ("label", "output", "prediction", "cyberbullying_type"):
        if key in payload and isinstance(payload[key], str):
            label = payload[key].strip().lower()
            return label if label in LABELS else "error"
    first_value = next(iter(payload.values()))
    label = str(first_value).strip().lower() if first_value is not None else "error"
    return label if label in LABELS else "error"


def process_with_groq(messages):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content
    except Exception as exc:
        print(f"API Error: {exc}")
        return None


def format_examples(examples):
    lines = ["Use these examples:"]
    for tweet, label in examples:
        lines.append(f"Tweet: {tweet}")
        lines.append(f"Answer: {{\"label\": \"{label}\"}}")
    return "\n".join(lines)


def build_zero_shot_messages(tweet):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Tweet: {tweet}"},
    ]


def build_static_few_shot_messages(tweet):
    examples_text = format_examples(STATIC_FEW_SHOT_EXAMPLES)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{examples_text}\n\nNow classify this tweet:\nTweet: {tweet}",
        },
    ]


def select_retrieval_examples(tweet, support_df, support_matrix, vectorizer, max_examples=6):
    query_vec = vectorizer.transform([tweet])
    similarities = cosine_similarity(query_vec, support_matrix).ravel()
    top_indices = np.argsort(similarities)[::-1][:120]

    selected = []
    covered_labels = set()

    for idx in top_indices:
        row = support_df.iloc[idx]
        label = str(row["cyberbullying_type"]).strip().lower()
        if label in LABELS and label not in covered_labels:
            selected.append((str(row["tweet_text"]), label))
            covered_labels.add(label)
        if len(selected) >= max_examples:
            break

    if len(selected) < max_examples:
        for idx in top_indices:
            row = support_df.iloc[idx]
            label = str(row["cyberbullying_type"]).strip().lower()
            if label in LABELS:
                candidate = (str(row["tweet_text"]), label)
                if candidate not in selected:
                    selected.append(candidate)
            if len(selected) >= max_examples:
                break

    return selected


def build_retrieval_few_shot_messages(tweet, support_df, support_matrix, vectorizer):
    examples = select_retrieval_examples(tweet, support_df, support_matrix, vectorizer, max_examples=6)
    examples_text = format_examples(examples)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{examples_text}\n\nNow classify this tweet:\nTweet: {tweet}",
        },
    ]


def run_method(method_name, tweets, message_builder):
    outputs = []
    predictions = []
    total = len(tweets)

    for idx, tweet in enumerate(tweets, start=1):
        messages = message_builder(str(tweet))
        result = process_with_groq(messages)
        outputs.append(result)
        predictions.append(extract_label(result))
        if idx % 20 == 0 or idx == total:
            print(f"{method_name}: processed {idx}/{total}")

    return outputs, predictions


eval_df = db.iloc[:SAMPLE_SIZE].copy()
support_df = db.iloc[SUPPORT_START : SUPPORT_START + SUPPORT_SIZE][["tweet_text", "cyberbullying_type"]].dropna().copy()

support_df["cyberbullying_type"] = support_df["cyberbullying_type"].astype(str).str.strip().str.lower()
support_df = support_df[support_df["cyberbullying_type"].isin(LABELS)]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
support_matrix = vectorizer.fit_transform(support_df["tweet_text"].astype(str).tolist())

eval_tweets = eval_df["tweet_text"].astype(str).tolist()
y_true = eval_df["cyberbullying_type"].astype(str).str.strip().str.lower().tolist()

print("Running zero-shot...")
_, y_pred_zero_shot = run_method("Zero-Shot", eval_tweets, build_zero_shot_messages)

print("Running static few-shot...")
_, y_pred_few_shot_static = run_method("Few-Shot (Static)", eval_tweets, build_static_few_shot_messages)

print("Running retrieval-based few-shot...")
_, y_pred_few_shot_retrieval = run_method(
    "Few-Shot (Retrieval TF-IDF)",
    eval_tweets,
    lambda tweet: build_retrieval_few_shot_messages(tweet, support_df, support_matrix, vectorizer),
)


def metric_row(name, y_true_labels, y_pred_labels):
    acc = accuracy_score(y_true_labels, y_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_labels,
        y_pred_labels,
        average="weighted",
        zero_division=0,
    )
    return {
        "Method": name,
        "Accuracy": acc,
        "Precision (Weighted)": precision,
        "Recall (Weighted)": recall,
        "F1 (Weighted)": f1,
    }


comparison_df = pd.DataFrame(
    [
        metric_row("Zero-Shot", y_true, y_pred_zero_shot),
        metric_row("Few-Shot (Static)", y_true, y_pred_few_shot_static),
        metric_row("Few-Shot (Retrieval TF-IDF)", y_true, y_pred_few_shot_retrieval),
    ]
)

print("\n================ PERFORMANCE COMPARISON ================")
print(comparison_df)

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
# select 1000 values from the dataset and create prompts
for chunk in tweets[0:100]:
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

def extract_label(response_text):
    if not isinstance(response_text, str):
        return "error"
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return "error"
    if not isinstance(payload, dict) or not payload:
        return "error"
    for key in ("label", "output", "prediction", "cyberbullying_type"):
        if key in payload and isinstance(payload[key], str):
            return payload[key].strip().lower()
    first_value = next(iter(payload.values()))
    return str(first_value).strip().lower() if first_value is not None else "error"

y_pred_zero_shot = [extract_label(item) for item in final_outputs]

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
# select 1000 values from the dataset and create prompts
# for chunk in tweets[0:1000]:
for chunk in tweets[0:100]:
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

#Improved label extraction by removing .split and using json parsing instead, also added error handling for non-string inputs and invalid JSON
y_pred_few_shot = [extract_label(item) for item in final_outputs]

import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

# 1. Get the true labels 
y_true = db['cyberbullying_type'][0:100].tolist()

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