# ⚔️ Model Arena

Side-by-side comparison and benchmarking of large language models, built with Streamlit.

Send one prompt to several models at once and compare their answers, latency and token usage. Or run a set of finance-flavoured benchmark tasks and see the results on a leaderboard.

## Features

| Tab | What it does |
| --- | --- |
| **⚔️ Arena** | Pick models, write a system prompt and a user prompt, set temperature and max tokens, and query all selected models in parallel. Responses appear side by side with latency, total tokens and tokens/second, plus bar charts. |
| **📊 Benchmark Suite** | Run built-in tasks (filterable by category) against selected models in parallel, with a progress bar. Results build up over the session. |
| **🏆 Leaderboard** | Per-model summary (runs, average and median latency, average tok/s, success rate, "speed wins" per task), average latency by category, a response explorer to read answers side by side, and CSV export. |

Supported backends:

- **Azure OpenAI**: API key, or Managed Identity when no key is set
- **Azure API Management (APIM)** in front of Azure OpenAI: subscription key, or Managed Identity when no key is set
- **Any OpenAI-compatible endpoint**: vLLM, Ollama (`/v1`), Mistral server, LM Studio, and others

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export MODEL_1_NAME="GPT-4o"
export MODEL_1_ID="gpt-4o"
export MODEL_1_ENDPOINT="https://<your-resource>.openai.azure.com"
export MODEL_1_API_KEY="<key>"
export MODEL_1_TYPE="azure"

export MODEL_2_NAME="Llama 3 (Ollama)"
export MODEL_2_ID="llama3"
export MODEL_2_ENDPOINT="http://localhost:11434/v1"
export MODEL_2_TYPE="openai_compat"

streamlit run app.py
```

Then open <http://localhost:8501>.

## Configuration

Models are configured through environment variables. There are up to **6 slots** (`N` = 1–6). A slot is enabled when `MODEL_N_NAME` is set.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `MODEL_N_NAME` | yes | – | Display name in the UI |
| `MODEL_N_ID` | | `gpt-4o` | Model name or Azure deployment name |
| `MODEL_N_ENDPOINT` | yes | – | Base URL of the API |
| `MODEL_N_API_KEY` | | – | API key or APIM subscription key. For `azure`/`apim`, leave it empty to use Managed Identity |
| `MODEL_N_TYPE` | | `apim` | `azure`, `apim` or `openai_compat` |
| `MODEL_N_API_VERSION` | | `2024-05-01-preview` | Azure OpenAI API version |
| `MODEL_N_COLOR` | | per slot | Hex colour for the model's card and charts |
| `MODEL_N_ICON` | | per slot | Emoji shown next to the model's name |

### Authentication

- **`azure`**: uses `MODEL_N_API_KEY` as the Azure OpenAI key.
- **`apim`**: sends the key both as the API key and in the `Ocp-Apim-Subscription-Key` header.
- **`azure` / `apim` without a key**: authenticates with `DefaultAzureCredential` (scope `https://cognitiveservices.azure.com/.default`). On Azure App Service this is the app's managed identity. Locally it falls back to Azure CLI login or environment credentials.
- **`openai_compat`**: the key is optional; `not-required` is sent if none is set.

### Demo fallback

If no `MODEL_N_NAME` is set, the app starts with two demo models:

| Model | Type | Variables |
| --- | --- | --- |
| GPT-4o (Azure) | `apim` | `APIM_ENDPOINT`, `APIM_SUBSCRIPTION_KEY`, `GPT4O_DEPLOYMENT` |
| Mistral (On-Prem) | `openai_compat` | `MISTRAL_ENDPOINT`, `MISTRAL_API_KEY`, `MISTRAL_MODEL_ID` |

## Benchmark tasks

Tasks are defined in [`benchmark_tasks.py`](benchmark_tasks.py). They run at temperature 0.3, each with its own token limit.

| Category | Tasks |
| --- | --- |
| Reasoning | Logical deduction, investment maths, transaction volume |
| Coding | Python CAGR function, SQL top customers, credit-risk classifier |
| Financial | Basel III summary, systematic vs unsystematic risk, mortgage rate impact |
| Summarization | ECB rate decision, AI strategy extract |
| Instruction Following | Exact numbered list, JSON self-description |
| Dutch Language | Mortgage explanation, AI in banking |

To add a task, append a `BenchmarkTask` to `TASKS`:

```python
BenchmarkTask(
    id="F4", category="Financial", name="Liquidity Coverage Ratio",
    prompt="Explain the Liquidity Coverage Ratio in 3 sentences.",
    max_tokens=256,
),
```

To add a new category, also add its name to `CATEGORIES` so it appears in the filter.

> **Note:** The leaderboard measures **speed and reliability** (latency, throughput, success rate), not answer quality. Use the response explorer to judge quality yourself.

Results live in the Streamlit session only. Export them as CSV from the Leaderboard tab before closing the browser tab.

## Deploying to Azure App Service

1. Create a Linux Web App with a Python 3.10+ runtime and deploy this repository.
2. Under **Configuration → Application settings**, add the `MODEL_N_*` variables.
3. Under **Configuration → General settings → Startup Command**, enter:
   ```
   bash startup.sh
   ```
   This starts Streamlit on `$PORT` (default `8000`), listening on all interfaces.
4. Optional: to avoid storing keys, turn on the app's **system-assigned managed identity**, grant it the *Cognitive Services OpenAI User* role on the Azure OpenAI resource, and leave `MODEL_N_API_KEY` empty.

## Project structure

```
.
├── app.py              # Streamlit UI: Arena, Benchmark Suite, Leaderboard
├── benchmark_tasks.py  # Benchmark task definitions and categories
├── config.py           # Loads model slots from environment variables
├── model_client.py     # Shared client for Azure, APIM and OpenAI-compatible APIs
├── requirements.txt
└── startup.sh          # Azure App Service startup command
```
