"""
Model Arena — LLM comparison and benchmarking
Rabobank GDAP · Future Fit Day
"""
import concurrent.futures
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from benchmark_tasks import CATEGORIES, TASKS, get_tasks_by_category
from config import ModelConfig, load_models
from model_client import ModelResponse, call_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Arena",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .arena-title { font-size:2rem; font-weight:800; color:#F08500; letter-spacing:-1px; margin:0; }
  .arena-sub   { color:#888; margin-top:-4px; margin-bottom:4px; }
  .model-card  {
    border-top: 4px solid var(--card-color, #ccc);
    border-radius: 0 0 8px 8px;
    background: #fafafa;
    padding: 10px 14px;
    margin-bottom: 6px;
  }
  .model-card .name { font-weight:700; font-size:1rem; }
  .model-card .meta { font-size:0.72rem; color:#777; margin-top:2px; }
  div.stButton > button[data-testid="baseButton-primary"] {
    background-color: #F08500;
    border: none;
  }
  div.stButton > button[data-testid="baseButton-primary"]:hover {
    background-color: #d97200;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "arena_results": [],   # list of (ModelConfig, ModelResponse)
    "arena_prompt":  "",
    "bench_results": [],   # list of result dicts accumulated across runs
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load models ───────────────────────────────────────────────────────────────
ALL_MODELS: list[ModelConfig] = load_models()
MODEL_MAP = {m.key: m for m in ALL_MODELS}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚔️ Model Arena")
    st.caption("Rabobank GDAP · Future Fit Day")
    st.divider()

    if ALL_MODELS:
        st.markdown("**Registered models**")
        for m in ALL_MODELS:
            st.markdown(
                f'<span style="color:{m.color}; font-size:1.1rem">●</span> '
                f'**{m.display_name}**<br>'
                f'<span style="font-size:0.7rem; color:#999">{m.api_type} · {m.model_id}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")
    else:
        st.warning("No models configured.")

    st.divider()
    with st.expander("ℹ️ Setup (env vars)"):
        st.code(
            "# Azure App Settings — repeat for MODEL_2, 3…\n"
            "MODEL_1_NAME=GPT-4o\n"
            "MODEL_1_ID=gpt-4o\n"
            "MODEL_1_ENDPOINT=https://apim.../openai\n"
            "MODEL_1_API_KEY=<subscription-key>\n"
            "MODEL_1_TYPE=apim\n\n"
            "MODEL_2_NAME=Mistral (On-Prem)\n"
            "MODEL_2_ID=mistral-large\n"
            "MODEL_2_ENDPOINT=http://mistral:8000/v1\n"
            "MODEL_2_API_KEY=not-required\n"
            "MODEL_2_TYPE=openai_compat",
            language="bash",
        )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="arena-title">⚔️ Model Arena</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="arena-sub">Side-by-side LLM comparison · Rabobank GDAP Future Fit Day</p>',
    unsafe_allow_html=True,
)
st.divider()

if not ALL_MODELS:
    st.error("No models configured. Set MODEL_N_* environment variables.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_arena, tab_bench, tab_leader = st.tabs(
    ["⚔️  Arena", "📊  Benchmark Suite", "🏆  Leaderboard"]
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _model_card_html(cfg: ModelConfig, resp: ModelResponse) -> str:
    tps = (
        round(resp.completion_tokens * 1000 / resp.latency_ms, 1)
        if resp.latency_ms else 0
    )
    status = "✅" if resp.success else "❌"
    return (
        f'<div class="model-card" style="--card-color:{cfg.color}">'
        f'<div class="name">{cfg.icon} {cfg.display_name}</div>'
        f'<div class="meta">'
        f'{status} &nbsp;⏱ {resp.latency_ms:,} ms &nbsp;|&nbsp; '
        f'📊 {resp.total_tokens} tok &nbsp;|&nbsp; ⚡ {tps} tok/s'
        f'</div></div>'
    )


def _bar_chart(
    x_vals: list,
    y_labels: list,
    colors: list,
    title: str,
    x_title: str,
    fmt_fn=str,
) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=x_vals, y=y_labels, orientation="h",
        marker_color=colors,
        text=[fmt_fn(v) for v in x_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, xaxis_title=x_title,
        height=240, margin=dict(l=0, r=70, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


def _run_parallel(
    model_list: list[ModelConfig],
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    max_workers: int = 8,
) -> list[tuple[ModelConfig, ModelResponse]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        fmap = {
            ex.submit(call_model, m, messages, temperature, max_tokens): m
            for m in model_list
        }
        raw = {fmap[f]: f.result() for f in concurrent.futures.as_completed(fmap)}
    return [(m, raw[m]) for m in model_list if m in raw]


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — ARENA
# ─────────────────────────────────────────────────────────────────────────────
with tab_arena:
    # Model selection row
    st.markdown("##### Select models to compare")
    chk_cols = st.columns(min(len(ALL_MODELS), 6))
    selected_models: list[ModelConfig] = []
    for col, m in zip(chk_cols, ALL_MODELS):
        with col:
            if st.checkbox(f"{m.icon} {m.display_name}", value=True, key=f"a_{m.key}"):
                selected_models.append(m)

    st.markdown("---")

    # Prompt inputs
    input_col, param_col = st.columns([4, 1])
    with input_col:
        with st.expander("⚙️ System prompt"):
            sys_prompt = st.text_area(
                "sys", value="You are a helpful, concise assistant.",
                height=68, label_visibility="collapsed",
            )
        user_prompt = st.text_area(
            "💬 Prompt",
            placeholder="Type your prompt here — e.g. 'Explain quantum computing in one paragraph'",
            height=130,
        )
    with param_col:
        st.markdown("**Parameters**")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05, key="a_temp")
        max_tokens  = st.slider("Max tokens",  64, 2048, 768, 64,   key="a_maxt")
        st.markdown("")
        run_btn = st.button(
            "▶ Run",
            type="primary",
            use_container_width=True,
            disabled=not selected_models or not user_prompt.strip(),
        )

    # Execute
    if run_btn and user_prompt.strip() and selected_models:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        with st.spinner(f"Querying {len(selected_models)} model(s) in parallel…"):
            results = _run_parallel(selected_models, messages, temperature, max_tokens)
        st.session_state.arena_results = results
        st.session_state.arena_prompt  = user_prompt

    # Display results
    if st.session_state.arena_results:
        results: list[tuple[ModelConfig, ModelResponse]] = st.session_state.arena_results
        st.markdown("---")
        st.caption(f"**Prompt:** {st.session_state.arena_prompt}")

        resp_cols = st.columns(len(results))
        for col, (cfg, resp) in zip(resp_cols, results):
            with col:
                st.markdown(_model_card_html(cfg, resp), unsafe_allow_html=True)
                if resp.success:
                    st.markdown(resp.content)
                else:
                    st.error(resp.error)

        # Metric charts
        st.markdown("---")
        names  = [cfg.display_name    for cfg, _    in results]
        colors = [cfg.color           for cfg, _    in results]
        lats   = [resp.latency_ms     for _,   resp in results]
        toks   = [resp.total_tokens   for _,   resp in results]

        mc1, mc2 = st.columns(2)
        with mc1:
            st.plotly_chart(
                _bar_chart(lats, names, colors, "⏱ Latency (ms)", "ms",
                           fmt_fn=lambda v: f"{v:,} ms"),
                use_container_width=True,
            )
        with mc2:
            st.plotly_chart(
                _bar_chart(toks, names, colors, "📊 Total Tokens", "tokens"),
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BENCHMARK SUITE
# ─────────────────────────────────────────────────────────────────────────────
with tab_bench:
    # ── Step 1: category filter (horizontal radio — always visible) ───────
    st.markdown("**Filter by category**")
    cat = st.radio(
        "Category",
        ["All"] + CATEGORIES,
        horizontal=True,
        label_visibility="collapsed",
    )
    filtered_tasks = get_tasks_by_category(cat)

    st.markdown("---")
    bcol_tasks, bcol_models = st.columns([3, 1])

    with bcol_tasks:
        # ── Step 2: task multiselect, defaulting to all tasks in the category
        task_options = [f"{t.name}  [{t.category} · {t.id}]" for t in filtered_tasks]
        task_by_label = {
            f"{t.name}  [{t.category} · {t.id}]": t for t in filtered_tasks
        }
        chosen_labels = st.multiselect(
            f"Tasks — {len(filtered_tasks)} available",
            options=task_options,
            default=task_options,       # all selected by default
            placeholder="Select one or more tasks…",
        )
        chosen_tasks = [task_by_label[l] for l in chosen_labels]

    with bcol_models:
        st.markdown("**Models**")
        bench_models: list[ModelConfig] = []
        for m in ALL_MODELS:
            if st.checkbox(f"{m.icon} {m.display_name}", value=True, key=f"b_{m.key}"):
                bench_models.append(m)
        st.markdown("")
        total_calls = len(chosen_tasks) * len(bench_models)
        bench_btn = st.button(
            f"▶ Run  ({total_calls} calls)",
            type="primary", use_container_width=True,
            disabled=not chosen_tasks or not bench_models,
        )

    if bench_btn and chosen_tasks and bench_models:
        prog   = st.progress(0.0, text="Starting…")
        done   = 0
        rows   = []

        def _bench_call(task, model):
            msgs = [
                {"role": "system", "content": task.system_prompt},
                {"role": "user",   "content": task.prompt},
            ]
            return call_model(model, msgs, temperature=0.3, max_tokens=task.max_tokens)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            fmeta = {
                ex.submit(_bench_call, task, model): (task, model)
                for task in chosen_tasks
                for model in bench_models
            }
            for fut in concurrent.futures.as_completed(fmeta):
                task, model = fmeta[fut]
                resp = fut.result()
                done += 1
                prog.progress(
                    done / total_calls,
                    text=f"{done}/{total_calls} — {task.name} × {model.display_name}",
                )
                tps = (
                    round(resp.completion_tokens * 1000 / resp.latency_ms, 1)
                    if resp.latency_ms else 0
                )
                preview = (
                    resp.content[:200] + "…"
                    if resp.success and len(resp.content) > 200
                    else resp.content
                )
                rows.append({
                    "timestamp":  datetime.now().strftime("%H:%M:%S"),
                    "task_id":    task.id,
                    "task_name":  task.name,
                    "category":   task.category,
                    "model_key":  model.key,
                    "model_name": model.display_name,
                    "latency_ms": resp.latency_ms,
                    "tokens":     resp.total_tokens,
                    "tok_per_s":  tps,
                    "success":    resp.success,
                    "response":   preview,
                    "error":      resp.error or "",
                })

        prog.empty()
        st.session_state.bench_results.extend(rows)
        st.success(f"✅ {len(rows)} results added. See the Leaderboard tab.")

    # Results preview
    if st.session_state.bench_results:
        st.markdown("---")
        df = pd.DataFrame(st.session_state.bench_results)
        df_view = df if cat == "All" else df[df["category"] == cat]

        if not df_view.empty:
            st.markdown(f"**Session results — {len(df_view)} runs**")
            st.dataframe(
                df_view[[
                    "task_name", "category", "model_name",
                    "latency_ms", "tokens", "tok_per_s", "success",
                ]].rename(columns={
                    "task_name":  "Task",
                    "category":   "Category",
                    "model_name": "Model",
                    "latency_ms": "Latency (ms)",
                    "tokens":     "Tokens",
                    "tok_per_s":  "Tok/s",
                    "success":    "OK",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Quick comparison charts
            avg = (
                df_view.groupby("model_name")[["latency_ms", "tok_per_s"]]
                .mean()
                .reset_index()
            )
            color_map = {m.display_name: m.color for m in ALL_MODELS}
            clrs = [color_map.get(n, "#999") for n in avg["model_name"]]

            cc1, cc2 = st.columns(2)
            with cc1:
                st.plotly_chart(
                    _bar_chart(
                        avg["latency_ms"].tolist(),
                        avg["model_name"].tolist(),
                        clrs,
                        "Avg Latency (ms)", "ms",
                        fmt_fn=lambda v: f"{v:.0f} ms",
                    ),
                    use_container_width=True,
                )
            with cc2:
                st.plotly_chart(
                    _bar_chart(
                        avg["tok_per_s"].tolist(),
                        avg["model_name"].tolist(),
                        clrs,
                        "Avg Throughput (tok/s)", "tok/s",
                        fmt_fn=lambda v: f"{v:.1f}",
                    ),
                    use_container_width=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_leader:
    if not st.session_state.bench_results:
        st.info("🏁 Run the Benchmark Suite to populate the leaderboard.")
    else:
        df_all = pd.DataFrame(st.session_state.bench_results)
        color_map = {m.display_name: m.color for m in ALL_MODELS}

        # ── Summary table ──────────────────────────────────────────────────
        summary = (
            df_all.groupby("model_name")
            .agg(
                Runs          = ("latency_ms",  "count"),
                Avg_Lat       = ("latency_ms",  "mean"),
                Median_Lat    = ("latency_ms",  "median"),
                Avg_TPS       = ("tok_per_s",   "mean"),
                Success_Rate  = ("success",     "mean"),
            )
            .reset_index()
            .rename(columns={"model_name": "Model"})
        )
        summary["Avg Latency (ms)"]    = summary["Avg_Lat"].round(0).astype(int)
        summary["Median Latency (ms)"] = summary["Median_Lat"].round(0).astype(int)
        summary["Avg Tok/s"]           = summary["Avg_TPS"].round(1)
        summary["Success Rate"]        = (summary["Success_Rate"] * 100).round(1).astype(str) + "%"

        # Speed wins — fastest model per task
        df_ok = df_all[df_all["success"]]
        if not df_ok.empty:
            fastest_idx  = df_ok.groupby("task_id")["latency_ms"].idxmin()
            wins_series  = df_all.loc[fastest_idx, "model_name"].value_counts()
            wins_df      = wins_series.reset_index()
            wins_df.columns = ["Model", "Speed Wins"]
            summary = summary.merge(wins_df, on="Model", how="left")
            summary["Speed Wins"] = summary["Speed Wins"].fillna(0).astype(int)
        else:
            summary["Speed Wins"] = 0

        summary = summary.sort_values("Avg Latency (ms)")[
            ["Model", "Runs", "Avg Latency (ms)", "Median Latency (ms)",
             "Avg Tok/s", "Success Rate", "Speed Wins"]
        ]

        st.markdown("#### 🏆 Overall Leaderboard")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # ── Category breakdown ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Performance by Category")
        cat_avg = (
            df_all.groupby(["category", "model_name"])["latency_ms"]
            .mean()
            .reset_index()
        )
        if not cat_avg.empty:
            fig = go.Figure()
            for model_name, grp in cat_avg.groupby("model_name"):
                fig.add_trace(go.Bar(
                    name=model_name,
                    x=grp["category"],
                    y=grp["latency_ms"].round(0),
                    marker_color=color_map.get(model_name, "#999"),
                    text=grp["latency_ms"].round(0).astype(int).astype(str) + " ms",
                    textposition="outside",
                ))
            fig.update_layout(
                barmode="group",
                title="Avg Latency (ms) by Category",
                yaxis_title="ms",
                height=380,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Response explorer ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Response Explorer")
        task_options = sorted(df_all["task_name"].unique())
        pick_task = st.selectbox("Select task to inspect", task_options)
        task_rows = df_all[df_all["task_name"] == pick_task]
        if not task_rows.empty:
            exp_cols = st.columns(min(len(task_rows), 4))
            for col, (_, row) in zip(exp_cols, task_rows.iterrows()):
                with col:
                    model_color = color_map.get(row["model_name"], "#ccc")
                    st.markdown(
                        f'<div style="border-top:4px solid {model_color}; '
                        f'padding:8px 12px; border-radius:0 0 6px 6px; background:#fafafa;">'
                        f'<b style="color:{model_color}">{row["model_name"]}</b><br>'
                        f'<span style="font-size:0.72rem; color:#777">'
                        f'⏱ {row["latency_ms"]:,} ms · 📊 {row["tokens"]} tok</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if row["success"]:
                        st.markdown(row["response"])
                    else:
                        st.error(row["error"])

        # ── Export / clear ─────────────────────────────────────────────────
        st.markdown("---")
        ex_col, clr_col = st.columns([3, 1])
        with ex_col:
            st.download_button(
                "⬇️ Export results (CSV)",
                data=df_all.to_csv(index=False),
                file_name=f"model_arena_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
        with clr_col:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.bench_results = []
                st.rerun()
