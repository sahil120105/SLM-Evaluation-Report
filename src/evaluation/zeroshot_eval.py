import os
import logging
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# ================================
# CONFIGURATION
# ================================

# ---- MODE SWITCH ----
TEST_MODE = True          # True = 10 samples, False = full dataset
TEST_SAMPLE_SIZE = 10

# ---- MODELS ----
EVAL_MODEL_NAME = "phi4-mini-reasoning:latest"
JUDGE_MODEL_NAME = "llama3.1:8b"

# ---- OLLAMA ----
OLLAMA_API_BASE = "http://localhost:11434/v1"

# ---- DATA ----
DATASET_PATH = "data/Golden_dataset/mdndocs012_golden_dataset.json"

# ================================
# ENVIRONMENT
# ================================

os.environ["MLFLOW_OPENAI_API_BASE"] = OLLAMA_API_BASE
os.environ["MLFLOW_OPENAI_API_KEY"] = "ollama"

# ================================
# LOGGING
# ================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ================================
# LOAD DATA
# ================================

logger.info(f"📂 Loading dataset: {DATASET_PATH}")

try:
    eval_data = pd.read_json(DATASET_PATH)
except ValueError:
    eval_data = pd.read_json(DATASET_PATH, lines=True)

eval_data = eval_data.rename(
    columns={"question": "inputs", "answer": "ground_truth"}
)

if TEST_MODE:
    logger.warning(
        f"🧪 TEST MODE ENABLED — using first {TEST_SAMPLE_SIZE} samples"
    )
    eval_data = eval_data.head(TEST_SAMPLE_SIZE).reset_index(drop=True)

logger.info(f"✅ Dataset size: {len(eval_data)} rows")

# ================================
# METRICS (STABLE SET ONLY)
# ================================

from mlflow.metrics import token_count
from mlflow.metrics.genai import answer_correctness

judge_prompt = (
    "You are a senior technical documentation reviewer. "
    "Compare the model answer to the reference answer. "
    "Score strictly on technical correctness. "
    "Penalize hallucinations, missing steps, or incorrect APIs."
)

correctness_metric = answer_correctness(
    model=f"openai:/{JUDGE_MODEL_NAME}",
    metric_metadata={"prompt": judge_prompt}
)

# ================================
# MODEL WRAPPER
# ================================

def ollama_model_wrapper(inputs: pd.DataFrame):
    client = OpenAI(
        base_url=OLLAMA_API_BASE,
        api_key="ollama"
    )

    predictions = []
    total = len(inputs)

    logger.info(f"🚀 Running inference on {total} questions")

    for idx, row in inputs.iterrows():
        question = row["inputs"]

        if idx % 5 == 0:
            logger.info(f"   Processing {idx + 1}/{total}")

        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a technical documentation assistant. "
                            "Answer concisely and accurately."
                        )
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.0,
                max_tokens=500
            )

            predictions.append(
                response.choices[0].message.content
            )

        except Exception as e:
            logger.error(f"❌ Error at row {idx}: {e}")
            predictions.append("ERROR")

    return predictions

# ================================
# MLFLOW EVALUATION
# ================================

mlflow.set_experiment("SLM_Closed_Book_Eval")

with mlflow.start_run(
    run_name=f"{'TEST' if TEST_MODE else 'FULL'}_{EVAL_MODEL_NAME}"
) as run:

    logger.info(f"🏁 MLflow Run ID: {run.info.run_id}")

    # ---- PARAMETERS ----
    mlflow.log_param("student_model", EVAL_MODEL_NAME)
    mlflow.log_param("judge_model", JUDGE_MODEL_NAME)
    mlflow.log_param("dataset_path", DATASET_PATH)
    mlflow.log_param("temperature", 0.0)
    mlflow.log_param("test_mode", TEST_MODE)
    mlflow.log_param("num_eval_samples", len(eval_data))

    # ---- RUN EVALUATION ----
    results = mlflow.models.evaluate(
        model=ollama_model_wrapper,
        data=eval_data,
        targets="ground_truth",
        model_type="text",
        extra_metrics=[
            token_count(),
            correctness_metric
        ],
    )

    # ============================
    # METRIC INSPECTION (SAFE)
    # ============================

    logger.info("📊 Logged metrics:")
    for k, v in results.metrics.items():
        logger.info(f"  {k}: {v}")

    avg_score = results.metrics.get(
        "mean_answer_correctness/v1/score"
    )

    if avg_score is None:
        logger.warning(
            "⚠️ answer_correctness metric not available."
        )
    else:
        logger.info(
            f"🏆 Average Correctness: {avg_score:.2f} / 5.0"
        )

    # ============================
    # ARTIFACTS
    # ============================

    eval_table = results.tables["eval_results_table"]

    csv_path = f"report_{run.info.run_id}.csv"
    eval_table.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    if "answer_correctness/v1/score" in eval_table.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            eval_table["answer_correctness/v1/score"],
            bins=5
        )
        plt.title(f"Correctness Score Distribution — {EVAL_MODEL_NAME}")
        plt.xlabel("Score (1–5)")
        plt.ylabel("Count")

        plot_path = "score_distribution.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

    logger.info("=" * 50)
    logger.info(
        f"📂 Artifacts: ./mlruns/"
        f"{run.info.experiment_id}/"
        f"{run.info.run_id}/artifacts"
    )
    logger.info("=" * 50)
