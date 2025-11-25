# Grounding

A Python project that uses Analysis of Competitive Hypotheses (ACH) via RAG as LLMs/AI grounding mechanism.

## Overview

LLMs hallucinate. Various strategies exist to cope with that. This project implements one of them.

The approach relies upon two pillars:

**First**, LLM statements are organized as a matrix of evidence vs. hypotheses, where each cell indicates the degree of hypothesis support by each piece of evidence.

| | Hypothesis1 | Hypothesis2 | ... | HypothesisN |
|---|---|---|---|---|
| Evidence1 | (positive, strong) | (positive, weak) | ... | (negative, weak) |
| Evidence2 | (neutral) | (negative, weak) | ... | (negative, strong) |
| ... | ... | ... | ... | ... |
| EvidenceK | (positive, strong) | (neutral) | ... | (negative, weak) |

This inherits from the Analysis of Competitive Hypotheses (ACH, https://en.wikipedia.org/wiki/Analysis_of_competing_hypotheses) approach. 

The benefits of this arrangement include:
- Stability against some fraction of rogue/wrong observations
- Stability against confirmation bias
- Cross-referencing of *each* piece of knowledge against *each* hypothesis, effectively "interrogating" all situational knowledge out of the LLM.
- Easy adoption of external evidence or hypotheses as further grounding constraints.


**Second**, LLMs only fill out the matrix, but the analysis of it is done via "classic" algorithms.

It is known (e.g. https://arxiv.org/abs/2401.11817, https://arxiv.org/pdf/2508.01781) that LLMs hallucinations on certain combinatorially complex yet practically important problems are inevitable and could not be arbitrarily reduced. Analysis of the ACH matrix above may require high combinatorial complexity. Thus, for reliable and repeatable answers that last leg of the analysis is done by the "classic" code.

## Output

The primary output of the evaluation system is likelihood scores of being true for each hypothesis. It is printed to the screen and saved as a .csv file to time-stamped folder in the in "Out/YYYY-MM-DD_HH-MM-SS/" folder.

Additionally, some intermediate outputs, including the statistical analysis of tokens costs, are saved to that folder, too.

## Process

The system operates through the following steps (all inputs and options are controlled via a single config file):
- Take pre-existing set of hypotheses and evidence from a user.
- Estimate approximate dollar cost of API call before execution, and ask for approval if the estimate exceeds $1.00
- (Optionally) generate new hypotheses about a given question/event
- (Optionally) collect and generate supporting and contradicting evidence
- Cross-validate each hypothesis against each evidence to build a support matrix
- Perform statistical analysis of the results

Internally, the project utilizes basic **Retrieval-Augmented Generation (RAG)** to enhance AI responses by providing context from previously generated hypotheses and evidence. This approach allows the system to build upon its own outputs iteratively, creating more coherent and contextually aware results.

## Approach

The methodology behind this system is designed to be **generic and flexible**, making it suitable for use by:
- **Humans**: Analysts can manually provide hypotheses and evidence through the configuration files
- **AI**: The system can autonomously generate hypotheses and evidence using language models
- **Hybrid workflows**: A combination of human-provided and AI-generated content

**Grounded Results**: The system supports grounding to any desired degree by allowing users to provide pre-existing, known hypotheses and evidence through the `ExtraH` and `ExtraE` configuration parameters. This enables:
- Starting with verified facts or expert-provided hypotheses
- Constraining AI generation within a known knowledge base
- Gradually building from human-curated content to AI-generated extensions
- Ensuring outputs are anchored to trusted sources when needed

## Features

- **Configurable Prompt Templates**: Multiple prompt styles (formal, semiformal, etc.)
- **Cost Estimation**: Pre-execution cost estimation to avoid unexpected charges
- **Telemetry**: Token usage tracking and statistical comparison
- **Flexible Configuration**: Key-value config files with support for strings, numbers, lists, and dictionaries
- **Statistical Analysis**: MAD (Median Absolute Deviation), t-tests, and Brunner-Munzel tests

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Grounding
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or on Windows:
```cmd
set OPENAI_API_KEY=your-api-key-here
```

## Configuration

The project uses configuration files (e.g., `Config-Pro.txt`) with the following format:

```
model = gpt-5-nano
CurrentPromptStyle = "question-first-semiformal"
Question = What profession or way of earning income will be highly valued by 2035?
nH = 1
ExtraH = {"Fast coder":False, "People manager":}
nE = 0
ExtraE = ["LLMs have fundamental difficulty with learning combinatorically complex patterns"]
Moniker = "Professions"
nMaxHLen = 128
nMaxELen = 280
extra_return_format = "Return ONLY profession name or very brief description of its nature, nothing else."
```

### Configuration Parameters

- `model`: OpenAI model to use (e.g., "gpt-5-nano", "gpt-5-mini")
- `CurrentPromptStyle`: Prompt template style (see `PromptTemplates.py`)
- `Question`: The main question to obtain the answer for
- `nH`: Number of new hypotheses to generate
- `ExtraH`: Dictionary of pre-existing hypotheses (keys are hypothesis text, optional values are booleans if the truthfulness of a hypothesis is known)
- `nE`: Number of evidence items per hypothesis (for both supports and contradicts) to seek/generated
- `ExtraE`: List of pre-existing evidence items
- `Moniker`: Identifier for the run (used in output filenames)
- `nMaxHLen`: Maximum length for hypotheses (in characters)
- `nMaxELen`: Maximum length for evidence (in characters)
- `extra_return_format`: Custom format instruction for AI-generated hypotheses
- `b_fill_matrix` (optional): Whether to fill the cross-reference matrix (default: True)

## Usage

Run the main script:

```bash
python Main.py
```

The script will:
1. Parse the configuration file (`Config-Pro.txt` by default)
2. Estimate the API cost and prompt for confirmation if cost > $1.00
3. Generate hypotheses and evidence
4. Build the support matrix
5. Save results to timestamped output folder in `Out/`

## Project Structure

```
.
├── Main.py                 # Main execution script
├── ConfigParser.py         # Configuration file parser
├── PromptTemplates.py      # Prompt template definitions
├── Telemetry.py           # Statistical analysis and cost estimation
├── Evaluator.py           # ACH evaluation logic
├── Consts.py              # Constants and shared utilities
├── Config-Pro.txt         # Example configuration file
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Key Functions

- `get_matrix()`: Generates hypotheses, evidence, and builds the support matrix
- `ask_h()`: Generates a new hypothesis via AI API call
- `ask_e()`: Generates supporting or contradicting evidence via AI API call
- `cross_ref()`: Determines support degree between hypothesis and evidence via AI API call
- `estimate_get_matrix_cost()`: Estimates API cost before execution
- `compare_lists_with_plots()`: Statistical comparison of call costs with visualization

## License

MIT License
Copyright (c) 2025 by Eugene V. Bobukh

## Contribution

Feel free to fork and use. But please mention Eugene V. Bobukh as the original code author if you are forking or using this code.