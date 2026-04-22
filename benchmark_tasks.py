"""
Pre-defined benchmark tasks across categories relevant to Rabobank.
Add tasks by appending to TASKS; categories auto-populate in the UI.
"""
from dataclasses import dataclass, field


@dataclass
class BenchmarkTask:
    id: str
    category: str
    name: str
    prompt: str
    system_prompt: str = "You are a helpful, concise assistant."
    max_tokens: int = 512


CATEGORIES = [
    "Reasoning",
    "Coding",
    "Financial",
    "Summarization",
    "Instruction Following",
    "Dutch Language",
]

TASKS: list[BenchmarkTask] = [
    # ── Reasoning ────────────────────────────────────────────────────────
    BenchmarkTask(
        id="R1", category="Reasoning", name="Logical Deduction",
        prompt=(
            "If all roses are flowers, and some flowers fade quickly, "
            "can we conclude that some roses fade quickly? "
            "Explain your reasoning step by step."
        ),
    ),
    BenchmarkTask(
        id="R2", category="Reasoning", name="Investment Maths",
        prompt=(
            "A portfolio starts at €100,000. It gains 20% in year 1, "
            "loses 15% in year 2, then gains 10% in year 3. "
            "What is the final value? Show all calculations."
        ),
    ),
    BenchmarkTask(
        id="R3", category="Reasoning", name="Transaction Volume",
        prompt=(
            "A bank processes 1,200 transactions per hour normally. "
            "During peak hours it processes 40% more. "
            "How many transactions in a 3-hour peak window? "
            "Show your work."
        ),
        max_tokens=256,
    ),

    # ── Coding ────────────────────────────────────────────────────────────
    BenchmarkTask(
        id="C1", category="Coding", name="Python CAGR",
        prompt=(
            "Write a Python function `cagr(start: float, end: float, years: int) -> float` "
            "that returns the Compound Annual Growth Rate as a percentage rounded to 2 dp. "
            "Include a one-line docstring."
        ),
        max_tokens=300,
    ),
    BenchmarkTask(
        id="C2", category="Coding", name="SQL Top Customers",
        prompt=(
            "Write a SQL query to find the top 5 customers by total transaction amount "
            "in the last 30 days. Table: transactions(id, customer_id, amount, created_at). "
            "Add brief inline comments."
        ),
        max_tokens=350,
    ),
    BenchmarkTask(
        id="C3", category="Coding", name="Credit Risk Classifier",
        prompt=(
            "Write a Python function that takes a credit score (300–850) and returns a "
            "risk category: 'Very Low' (750+), 'Low' (700–749), 'Medium' (650–699), "
            "'High' (600–649), 'Very High' (<600). Keep it clean and readable."
        ),
        max_tokens=250,
    ),

    # ── Financial ────────────────────────────────────────────────────────
    BenchmarkTask(
        id="F1", category="Financial", name="Basel III Summary",
        prompt=(
            "Explain Basel III capital requirements in exactly 3 bullet points "
            "for a non-technical bank employee."
        ),
        max_tokens=300,
    ),
    BenchmarkTask(
        id="F2", category="Financial", name="Systematic vs Unsystematic Risk",
        prompt=(
            "Explain the difference between systematic and unsystematic risk. "
            "Give one concrete portfolio example of each."
        ),
        max_tokens=350,
    ),
    BenchmarkTask(
        id="F3", category="Financial", name="Mortgage Rate Impact",
        prompt=(
            "A customer has a €300,000 mortgage at 2% interest over 30 years. "
            "If rates rise to 5%, how does their monthly payment change? "
            "Explain the impact in plain language."
        ),
        max_tokens=400,
    ),

    # ── Summarization ────────────────────────────────────────────────────
    BenchmarkTask(
        id="S1", category="Summarization", name="ECB Rate Decision",
        prompt=(
            "Summarise the following in 2–3 sentences:\n\n"
            "The European Central Bank raised its key interest rates by 25 basis points at "
            "its June meeting, marking the eighth consecutive hike as policymakers battle "
            "elevated inflation. The deposit facility rate now stands at 3.5%, its highest "
            "level since 2001. ECB President Christine Lagarde signalled further tightening "
            "was likely unless incoming data showed a significant shift in the inflation "
            "outlook. Core inflation remained stubbornly above 5%, well above the 2% target, "
            "despite headline inflation falling from its peak of over 10% last year. "
            "Financial markets had largely priced in the move but reacted to the hawkish "
            "tone by pushing bond yields higher across the eurozone."
        ),
        max_tokens=200,
    ),
    BenchmarkTask(
        id="S2", category="Summarization", name="AI Strategy Extract",
        prompt=(
            "Summarise the key takeaway from this paragraph in one sentence:\n\n"
            "Many European banks are investing heavily in generative AI to automate "
            "back-office processes, accelerate credit decisioning, and improve customer "
            "engagement through conversational interfaces. While productivity gains are "
            "promising, risk teams are concerned about model hallucinations, data privacy "
            "under GDPR, and explainability requirements under EU AI Act obligations. "
            "Successful deployments share a common pattern: tight human-in-the-loop "
            "oversight, rigorous red-teaming, and clear escalation paths."
        ),
        max_tokens=100,
    ),

    # ── Instruction Following ────────────────────────────────────────────
    BenchmarkTask(
        id="I1", category="Instruction Following", name="Exact Numbered List",
        prompt=(
            "List EXACTLY 3 benefits of AI in banking. Use this format:\n"
            "1. [Benefit]: [One sentence explanation]\n"
            "2. [Benefit]: [One sentence explanation]\n"
            "3. [Benefit]: [One sentence explanation]\n"
            "Do not add any text before or after the list."
        ),
        max_tokens=200,
    ),
    BenchmarkTask(
        id="I2", category="Instruction Following", name="JSON Self-Description",
        prompt=(
            'Return a JSON object with exactly these fields: '
            '{"model_strength": "...", "best_use_case": "...", "limitation": "..."}. '
            "Fill each with a brief description of yourself as an AI model. "
            "Return only valid JSON — no markdown fences, no extra text."
        ),
        max_tokens=200,
    ),

    # ── Dutch Language ───────────────────────────────────────────────────
    BenchmarkTask(
        id="NL1", category="Dutch Language", name="Hypotheek Uitleg",
        prompt=(
            "Leg in het Nederlands uit wat een annuïteitenhypotheek is en hoe de "
            "maandlasten zijn opgebouwd. Maximaal 4 zinnen, begrijpelijk voor een klant "
            "zonder financiële achtergrond."
        ),
        max_tokens=300,
    ),
    BenchmarkTask(
        id="NL2", category="Dutch Language", name="AI bij Rabobank",
        prompt=(
            "Beschrijf in het Nederlands in 3 zinnen wat de belangrijkste voordelen van "
            "kunstmatige intelligentie zijn voor een coöperatieve bank zoals Rabobank."
        ),
        max_tokens=250,
    ),
]

TASK_MAP = {t.id: t for t in TASKS}


def get_tasks_by_category(category: str) -> list[BenchmarkTask]:
    if category == "All":
        return TASKS
    return [t for t in TASKS if t.category == category]
