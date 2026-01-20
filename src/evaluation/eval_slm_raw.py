import mlflow
import pandas as pd
import os
from openai import OpenAI

# --- IMPORTS ---
from mlflow.metrics import rougeL, token_count
from mlflow.metrics.genai import (
    answer_correctness, 
    answer_similarity, 
    EvaluationExample
)

# --- Configuration ---
EVAL_MODEL_NAME = "phi4-mini-reasoning:latest" 
JUDGE_MODEL_NAME = "llama3.2:1b" 

# CRITICAL FIX: The base URL for the env var MUST include /v1 for metrics to work
OLLAMA_API_BASE = "http://localhost:11434/v1"

# Route MLflow's GenAI metrics to your local Ollama
os.environ["MLFLOW_OPENAI_API_BASE"] = OLLAMA_API_BASE
os.environ["MLFLOW_OPENAI_API_KEY"] = "ollama"

# --- 1. Load Data ---
file_path = "data/Golden_dataset/mdndocs012_golden_dataset.json"

try:
    eval_data = pd.read_json(file_path)
    print("✅ Detected Standard JSON format.")
except ValueError:
    eval_data = pd.read_json(file_path, lines=True)
    print("✅ Detected JSONL format.")

eval_data = eval_data.rename(columns={
    "question": "inputs", 
    "answer": "ground_truth"
})

# --- 2. Setup Tracking ---
mlflow.openai.autolog()
print("✅ MLflow OpenAI autologging enabled.")

# Use local file storage (mlruns) if no server is running, or set your URI
# mlflow.set_tracking_uri("http://localhost:5000") 
mlflow.set_experiment("Ollama_Eval")
print("✅ MLflow tracking set up.")

# --- 3. Define Model Wrapper (The Student) ---
def ollama_model_wrapper(inputs):
    # Initialize client inside wrapper to ensure thread safety during eval
    client = OpenAI(
        base_url=OLLAMA_API_BASE,
        api_key="ollama",
    )
    predictions = []
    
    # inputs is a DataFrame, iterate over the 'inputs' column
    for question in inputs["inputs"]:
        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=500
            )
            predictions.append(response.choices[0].message.content)
        except Exception as e:
            predictions.append(f"Error: {str(e)}")
            
    return predictions

# --- 4. Define Metrics (The Judge) ---
similarity_metric = answer_similarity(model="sentence-transformers/all-MiniLM-L6-v2")

example_obj = EvaluationExample(
    input="How do I check the version?",
    output="Run git --version",
    score=5,
    justification="The answer is exact and correct."
)

correctness_metric = answer_correctness(
    model=f"openai:/{JUDGE_MODEL_NAME}",
    examples=[example_obj]
)

# --- 5. Run Evaluation ---
print(f"🚀 Starting evaluation for model: {EVAL_MODEL_NAME}")

with mlflow.start_run(run_name=f"Eval_{EVAL_MODEL_NAME}"):
    # 1. Smoke Test (Your single request) - This will be autologged
    print("running smoke test...")
    client = OpenAI(base_url=OLLAMA_API_BASE, api_key="ollama")
    client.chat.completions.create(
        model=EVAL_MODEL_NAME, # Changed to use your eval model
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
        max_tokens=100,
    )

    # 2. Full Dataset Evaluation
    print("Running full dataset evaluation...")
    results = mlflow.models.evaluate(
        model=ollama_model_wrapper,
        data=eval_data,
        targets="ground_truth",
        model_type="text",
        extra_metrics=[rougeL(), token_count(), similarity_metric, correctness_metric],
    )

    print(f"\n📊 Metrics for {EVAL_MODEL_NAME}:")
    print(results.metrics)
    
    # Save results locally
    output_filename = f"results_{EVAL_MODEL_NAME.replace(':', '_')}.csv"
    results.tables["eval_results_table"].to_csv(output_filename)
    print(f"\n✅ Results saved to {output_filename}")