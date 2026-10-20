#!/usr/bin/env python3
"""
interpret_results.py
====================
Anthropic SDK agent that reads the pipeline's REPORT.md and results.json,
then writes a rigorous scientific commentary (COMMENTARY.md) interpreting
the findings in the context of Born rule verification and GPT rank theory.

The agent is given two file-access tools so it can autonomously load the
pipeline output before reasoning. It uses Claude Opus 4.6 with adaptive
thinking for deep scientific analysis.

Usage
-----
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

    python interpret_results.py                  # reads results/ by default
    python interpret_results.py --results results/   # explicit path

Workflow
--------
    1. Claude reads REPORT.md  (chi²/dof tables, hypothesis verdict)
    2. Claude reads results.json  (raw K-sweep data per N_eff)
    3. Claude reasons about the physics (adaptive thinking enabled)
    4. Claude writes COMMENTARY.md via the write_commentary tool
"""

import argparse
from pathlib import Path

import anthropic

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL      = "claude-opus-4-6"
MAX_TOKENS = 8192

# ── Tool definitions ──────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read the full text contents of a file on disk. "
            "Use this to load REPORT.md and results.json from the "
            "pipeline output directory before writing your commentary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_commentary",
        "description": (
            "Write the finished scientific commentary to COMMENTARY.md. "
            "Call this exactly once, after you have read both data files "
            "and completed your full analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Output path for COMMENTARY.md.",
                },
                "content": {
                    "type": "string",
                    "description": "Full markdown content of the scientific commentary.",
                },
            },
            "required": ["path", "content"],
        },
    },
]


# ── Tool execution ────────────────────────────────────────────────────────────
def _execute_tool(name: str, tool_input: dict) -> str:
    if name == "read_file":
        path = Path(tool_input["path"])
        if not path.exists():
            return f"Error: file not found — {path}"
        return path.read_text(encoding="utf-8")

    if name == "write_commentary":
        path = Path(tool_input["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tool_input["content"], encoding="utf-8")
        return f"Commentary written to {path}"

    return f"Error: unknown tool '{name}'"


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a quantum physicist specialising in Generalised Probabilistic Theories
(GPTs) and experimental foundations of quantum mechanics. You have access to
two tools: read_file and write_commentary.

Your task
---------
Step 1  Read REPORT.md to understand the pipeline's methodology and results.
Step 2  Read results.json to examine the raw cross-validated chi^2/dof data.
Step 3  Write a rigorous scientific commentary to COMMENTARY.md that covers:

    a) Context — place the findings within the GPT rank literature and the
       broader programme of experimental tests of the Born rule (Aaronson,
       Dakic & Brukner, Sorkin I_3 = 0, etc.).

    b) Chi^2/dof curve analysis — describe the shape of the curve for each
       slit group and N_eff. Where does the minimum occur, how sharp is the
       elbow, and what does this imply about the effective state-space dimension?

    c) Hypothesis assessment — does the data support H_0 (QM: rank = n^2 - 1 = 15)
       over H_1 (classical: rank = 4) and H_2 (2nd-order: rank approx 10)?
       Quantify using the reported Delta chi^2 between K=14 and K=15.

    d) Noise robustness — how does the rank signature change with N_eff
       (photon count)? At what N_eff does the minimum become unambiguous?

    e) Limitations — this pipeline uses only Poisson shot noise; real emICCD
       data will also contain dark counts, beam drift, and mode mismatch.
       Discuss what EXP_N_EFF_SCALE does conceptually and what systematic
       budget would be needed for a convincing experimental result.

    f) Next steps — gated SPDC source, single-photon coincidences, systematic
       error characterisation, and the path to a publishable result.

    g) Abstract — end with a 5-sentence abstract suitable for submission to
       Physical Review Letters or npj Quantum Information.

Style
-----
- Formal scientific prose (continuous paragraphs, not bullet points).
- Inline LaTeX for all mathematical expressions: $...$, $$...$$.
- Markdown headings (## and ###) for sections.
- Roughly 1 200 to 2 000 words in the main body.
- Do not reproduce raw tables from the report — refer to findings by value.
"""


# ── Main agent loop ───────────────────────────────────────────────────────────
def run(results_dir: str):
    client = anthropic.Anthropic()
    results = Path(results_dir)

    report_path     = results / "REPORT.md"
    json_path       = results / "results.json"
    commentary_path = results / "COMMENTARY.md"

    if not report_path.exists():
        print(f"[Interpreter] {report_path} not found.")
        print("[Interpreter] Run the pipeline first:  python run_experiment.py")
        return

    initial_prompt = (
        f"Please analyse the GPT rank pipeline output in `{results_dir}/`.\n\n"
        f"Files to read:\n"
        f"  • {report_path}  — full scientific report with chi^2/dof tables\n"
        f"  • {json_path}  — raw numerical results (K-sweep per N_eff and group)\n\n"
        f"Write the final commentary to: {commentary_path}\n\n"
        f"Read both files before you begin writing."
    )

    messages = [{"role": "user", "content": initial_prompt}]

    print(f"[Interpreter] Model      : {MODEL} (adaptive thinking)")
    print(f"[Interpreter] Results dir: {results_dir}")
    print(f"[Interpreter] Output     : {commentary_path}")
    print()

    # ── Agentic loop ──────────────────────────────────────────────────────────
    turn = 0
    while True:
        turn += 1
        print(f"[Interpreter] Turn {turn} — calling Claude …")

        with client.messages.stream(
            model      = MODEL,
            max_tokens = MAX_TOKENS,
            thinking   = {"type": "adaptive"},
            system     = SYSTEM_PROMPT,
            tools      = TOOLS,
            messages   = messages,
        ) as stream:
            response = stream.get_final_message()

        # Log any text the model produced this turn
        for block in response.content:
            if block.type == "text" and block.text.strip():
                preview = block.text[:300].replace("\n", " ")
                print(f"[Claude]       {preview}{'…' if len(block.text) > 300 else ''}")

        # ── Done ─────────────────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            print()
            print("[Interpreter] Agent finished.")
            if commentary_path.exists():
                size = commentary_path.stat().st_size
                print(f"[Interpreter] Commentary : {commentary_path}  ({size:,} bytes)")
            else:
                print("[Interpreter] Note: write_commentary was not called.")
            break

        # ── Tool use ──────────────────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                print(f"[Tool]         {block.name}({', '.join(f'{k}={repr(v)[:40]}' for k, v in block.input.items())})")
                result  = _execute_tool(block.name, block.input)
                preview = result[:120].replace("\n", " ")
                print(f"               → {preview}{'…' if len(result) > 120 else ''}")

                tool_results.append({
                    "type"       : "tool_result",
                    "tool_use_id": block.id,
                    "content"    : result,
                })

            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop
        print(f"[Interpreter] Unexpected stop_reason: {response.stop_reason}")
        break


def main():
    parser = argparse.ArgumentParser(
        description="Anthropic SDK agent — interpret GPT rank pipeline results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results", default="results",
        help="Pipeline output directory (must contain REPORT.md).",
    )
    args = parser.parse_args()
    run(args.results)


if __name__ == "__main__":
    main()
