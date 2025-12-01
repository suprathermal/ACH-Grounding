# prompt structure and forgotten system part
# More interpretable charts output
# Mention telemetry/logging is basic
# That we use only OpenAI
from openai import OpenAI

import os
import pandas as pd
from datetime import datetime
from tkinter import filedialog
import tkinter as tk

from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, List, Any

import Evaluator as ev
from PromptTemplates import PROMPT_TEMPLATES
from Consts import magic_split, g_t_supported_values
from Telemetry import compare_toks, estimate_get_matrix_cost, ValueWithErrToStr
from ConfigParser import parse_config

# Dictionaries to track token costs/use
d_l_e_type_costs = {}
l_d_tok = []

def get_config() -> Dict[str, Any]:
    """Prompt user to select a config file and return the parsed configuration."""
    # Prompt user to select config file
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    s_config = filedialog.askopenfilename(
        title="Select Config File",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()
    
    if not s_config:
        print("No config file selected. Exiting.")
        return None
    
    return parse_config(s_config)

def get_client() -> OpenAI:
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error getting client or API key: {e}")
        return None

def ask_h(client: OpenAI, s_event: str, l_h: List[str], d_config: Dict[str, Any], max_h_len: int, prompt_style: str, model_in: str = "gpt-5-mini") -> str:
    class Hypothesis(BaseModel):
        hypothesis: str = Field(max_length=max_h_len)

    s_h = "\n".join(l_h)
    templates = PROMPT_TEMPLATES[prompt_style]

    s_request = templates["hypothesis_request"].format(event=s_event)
    s_request += "\n" + templates["consideration"] + "\n"
    if (len(s_h) > 0):
        s_request += "\n\n" + templates["differentiation"] + "\n" + s_h

    # Get extra_return_format from config dictionary
    extra_return_format = d_config.get("extra_return_format")

    s_user_content = s_request + "\n"
    s_user_content += f"\n\n{templates['return_format'].format(extra_return_format=extra_return_format)} no longer than {max_h_len} characters."

    response = client.responses.parse(
        model=model_in,
        input=[
            {"role": "system", "content": templates["intro"]},
            {"role": "user", 
            "content": s_user_content}            
        ],
        text_format=Hypothesis
    )

    event = response.output_parsed

    l_d_tok.append({"type":"h", "cond_len":len(l_h), "mode":"", "event":s_event, 
    "input_tokens":response.usage.input_tokens, "output_tokens":response.usage.output_tokens,
    "reasoning_tokens":response.usage.output_tokens_details.reasoning_tokens})

    return event.hypothesis

def ask_e(client: OpenAI, s_h: str, s_rel: str, l_e: List[str], d_config: Dict[str, Any], max_e_len: int, prompt_style: str, model_in: str = "gpt-5-mini") -> str:
    class Evidence(BaseModel):
        ev: str = Field(max_length=max_e_len)

    s_e = "\n".join(l_e)
    templates = PROMPT_TEMPLATES[prompt_style]

    s_request = ""
    # Check if template expects {event} parameter
    if "{event}" in templates["evidence_request"]:
        s_request += templates["evidence_request"].format(relation=s_rel, hypothesis=s_h, event=d_config["Question"])
    else:
        s_request += templates["evidence_request"].format(relation=s_rel, hypothesis=s_h)
    s_request += "\n" + templates["evidence_effort"] + "\n"
    if (len(s_e) > 0):
        s_request += "\n\n" + templates["evidence_differentiation"] + "\n" + s_e

    s_user_content = s_request + "\n"
    s_user_content += f"\n\n{templates['evidence_format']} no longer than {max_e_len} characters."

    response = client.responses.parse(
        model=model_in,
        input=[
            {"role": "system", "content": templates["intro"]},
            {"role": "user", 
            "content": s_user_content}            
        ],
        text_format=Evidence
    )

    event = response.output_parsed

    if not (s_rel in d_l_e_type_costs.keys()):
        d_l_e_type_costs[s_rel] = []
    d_l_e_type_costs[s_rel].append(response.usage.total_tokens)

    l_d_tok.append({"type":"e", "cond_len":len(l_e), "mode":s_rel, "event":s_h, 
    "input_tokens":response.usage.input_tokens, "output_tokens":response.usage.output_tokens,
    "reasoning_tokens":response.usage.output_tokens_details.reasoning_tokens})

    return event.ev               

def cross_ref(client: OpenAI, s_h: str, s_e: str, d_config: Dict[str, Any], prompt_style: str, model_in: str = "gpt-5-mini") -> int:
    class SupportDegree(BaseModel):
        support_degree: Literal[*g_t_supported_values]

    templates = PROMPT_TEMPLATES[prompt_style]
    
    s_request = ""
    # Check if template expects {event} parameter
    if "{event}" in templates["support_request"]:
        s_request += templates["support_request"].format(evidence=s_e, hypothesis=s_h, event=d_config["Question"]) + " "
    else:
        s_request += templates["support_request"].format(evidence=s_e, hypothesis=s_h) + " "
    s_request += templates["support_scale"]

    s_user_content = s_request + "\n"
    s_user_content += "\n\n" + templates["support_format"]

    response = client.responses.parse(
        model=model_in,
        input=[
            {"role": "system", "content": templates["intro"]},
            {"role": "user", 
            "content": s_user_content}            
        ],
        text_format=SupportDegree
    )

    event = response.output_parsed

    support_degree = event.support_degree

    l_d_tok.append({"type":"r", "cond_len":1, "mode":str(support_degree), "event":s_h + magic_split + s_e, 
    "input_tokens":response.usage.input_tokens, "output_tokens":response.usage.output_tokens,
    "reasoning_tokens":response.usage.output_tokens_details.reasoning_tokens})

    return support_degree  

def get_user_cost_confirmation(estimated_cost: float) -> bool:
    """Prompt user to confirm if they want to proceed with the estimated cost.
    
    Returns True if user wants to proceed, False otherwise.
    """
    print(f"Estimated cost (2x error range): ${estimated_cost:.2f}")
    if (estimated_cost > 1.0):
        print("Proceed? Press 'y' to continue, any other key to exit")
        s_answer = input()
        if (s_answer != "y"):
            print("Exiting")
            return False
    return True

def get_matrix(client: OpenAI, d_config: Dict[str, Any]) -> pd.DataFrame:
    # Extract parameters from config
    s_event = d_config["Question"]
    nH = d_config["nH"]
    nE_per_H = d_config["nE"]
    max_h_len = d_config.get("nMaxHLen", 128)
    max_e_len = d_config.get("nMaxELen", 300)
    d_h_pre = d_config["ExtraH"]
    l_h_pre = list(d_h_pre.keys())
    l_e_pre = d_config["ExtraE"]
    model_in = d_config["model"]
    prompt_style = d_config["CurrentPromptStyle"]
    b_fill_matrix = d_config.get("b_fill_matrix", True)

    l_h = l_h_pre.copy()
    print("\n-----------Starting extra hypothesis collection---------------------")   
    for i in range(nH):
        l_h.append(ask_h(client, s_event, l_h, d_config, max_h_len, prompt_style, model_in=model_in))
        print(f"\rGenerated {i+1} entries out of {nH}", end='', flush=True)
    print("\n----------------------------------------------------------")
    print("Hypotheses:")
    print(l_h)

    l_e = l_e_pre.copy()
    print("\n-----------Starting extra evidence collection---------------------")       
    nC = 0
    for s_h in l_h:
        for i in range(nE_per_H):
            l_e.append(ask_e(client, s_h, "supports", l_e, d_config, max_e_len, prompt_style, model_in=model_in))
            l_e.append(ask_e(client, s_h, "contradicts", l_e, d_config, max_e_len, prompt_style, model_in=model_in))
            nC += 1
            print(f"\rGenerated {nC} entries out of {len(l_h) * nE_per_H}", end='', flush=True)
    print("\n----------------------------------------------------------")
    print("Evidence:")
    print(l_e)

    print("\n-----------Starting cross-referencing---------------------")   
    if b_fill_matrix:
        nC = 0
        l_d = []
        for e in range(len(l_e)):
            d = {"E": l_e[e]}
            for h in range(len(l_h)):            
                n_sup = cross_ref(client, l_h[h], l_e[e], d_config, prompt_style, model_in=model_in)
                d[l_h[h]] = n_sup
                nC += 1
                print(f"\rCross-referenced {nC} entries out of {len(l_e) * len(l_h)}", end='', flush=True)
            l_d.append(d)
        
        df = pd.DataFrame(l_d, columns=["E"] + l_h)
    else:
        df = pd.DataFrame(columns=["E"] + l_h)

    print("\n----------------------------------------------------------")

    return df

def main() -> None:
    print("Started")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    

    d_config = get_config()
    if (d_config is None): return

    s_moniker = d_config["Moniker"]
    s_model = d_config["model"]
    d_h_pre = d_config["ExtraH"]
    if "extra_return_format" not in d_config.keys() or d_config["extra_return_format"] is None:
        d_config["extra_return_format"] = "Return only ONE answer as a concise, clear sentence"

    client = get_client()        

    estimated_cost = estimate_get_matrix_cost(d_config)
    if not get_user_cost_confirmation(estimated_cost): return

    df = get_matrix(client = client, d_config = d_config)

    # Create timestamped output folder
    output_folder = f"Out/{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

    compare_toks("-" + s_moniker + "-" + s_model, d_h_pre, output_folder, l_d_tok)
    df.to_csv(f"{output_folder}/{s_moniker}-{s_model}.csv", index=False)

    df_eval = ev.eval_ACH_basic(df)
    df_eval.to_csv(f"{output_folder}/{s_moniker}-{s_model}-eval.csv", index=False)
    df_eval.apply(lambda r: print(f"Hypothesis: {r['hypothesis']}, score = {ValueWithErrToStr(r['avg'], r['stderr'])}"), axis=1)

    print("Finished")

if __name__ == "__main__":
    main()