# 📊 VARIANCE-LIES

[![CI](https://github.com/kapildhinwa2010-dev/VARIEANCE-LIES/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/VARIEANCE-LIES/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Variance captured ≠ behavior preserved

Compression reports love to say **"99% of activation variance preserved"**.
This probe answers: yes, but **how much of the behavior survived?**

Two curves, one rank sweep:

- 📈 `variance_captured` — how much delta energy the rank-k basis keeps
- 🧠 `behavior` — how many generated tokens still match the original model

If the curves disagree, the compression metric is **lying**.

## 🔍 What you typically see

- 📈 Variance climbs fast and looks great (90%+ at low rank)
- 🧠 Behavior lags far behind, or drops to zero after the first few tokens
- 🦋 A few agreeing tokens hide the collapse that follows

## ⚡ Install

GPU (default):

    pip install torch
    pip install -r requirements.txt

<details>
<summary>🖥️ CPU-only variant</summary>

    pip install --index-url https://download.pytorch.org/whl/cpu torch
    pip install -r requirements.txt

</details>

## 💻 Usage

    python variance_lies.py --model Qwen/Qwen2.5-0.5B-Instruct --ranks 4,8,16,32,64

Output: a table + `variance_vs_behavior.json` with both curves.

## 🧪 Tests

    pytest -q

## 🤔 Why this matters

If variance were behavior, compression would be solved. It is not.
This repo is the smallest honest instrument that shows the difference
on any Llama/Qwen/GPT2-style model, on a laptop GPU or CPU. 🙏

## ⚖️ License

MIT - see [LICENSE](LICENSE).
