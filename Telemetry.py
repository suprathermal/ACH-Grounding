"""
Utilities for telemetry and cost estimation.
Most of this code was written by AI.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict, Optional, Any, Union

from Consts import magic_split, d_API_cost_per_M

g_l_token_types_to_eval = ["input_tokens", "reasoning_tokens"]

def ValueWithErrToStr(fVal: Optional[float], fErr: Optional[float], nDigits: int = 3) -> str:
    if (fVal is None):
        if (fErr is None):
            return "NaN ± NaN"
        else:
            return  ("NaN ± {0:.3f}".format(fErr))
    else:
        if (fErr is None):
            return "{0:.3f} ± NaN".format(fVal)
        else:
            return "{0:.3f} ± {1:.3f}".format(fVal, fErr)

def calculate_mad(data: np.ndarray) -> float:
    """Calculate Median Absolute Deviation (MAD) for a dataset."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return mad

def compare_lists_with_plots(list1: List[float], list2: List[float], list1_label: str = "List 1", list2_label: str = "List 2", s_suffix: str = "", output_folder: str = "") -> Dict[str, Union[float, np.floating]]:
    """
    Compare two lists by computing median, MAD, p-value, and creating overlapping bar charts.
    
    Parameters:
    - list1, list2: Lists of numerical values to compare
    - list1_label, list2_label: Labels for the lists in plots
    - s_suffix: Suffix for output files
    - output_folder: Folder path for saving plots
    
    Returns:
    - Dictionary containing medians, MADs, and p-value
    """
    
    # Check if either list is empty - return all NaNs if so
    if len(list1) == 0 or len(list2) == 0:
        return {
            'median1': np.nan,
            'median2': np.nan,
            'mad1': np.nan,
            'mad2': np.nan,
            'p_value': np.nan,
            'median_difference': np.nan,
            'brunner_munzel_statistic': np.nan,
            'brunner_munzel_p_value': np.nan
        }

    # Convert to numpy arrays for easier computation
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    
    # Calculate medians
    median1 = np.median(arr1)
    median2 = np.median(arr2)
    
    # Calculate MADs
    mad1 = calculate_mad(arr1)
    mad2 = calculate_mad(arr2)
    
    # Calculate p-value using MAD as a robust measure of scale
    # We'll use a two-sample t-test but with MAD-based standardization
    # Convert MAD to standard deviation equivalent (MAD * 1.4826 ≈ std for normal distribution)
    mad1_std_equiv = mad1 * 1.4826 if mad1 > 0 else 1e-10
    mad2_std_equiv = mad2 * 1.4826 if mad2 > 0 else 1e-10
    
    # Perform two-sample t-test using the difference of medians
    # Since we're using MAD, we'll create a custom test statistic
    pooled_mad = np.sqrt((mad1_std_equiv**2 / len(arr1)) + (mad2_std_equiv**2 / len(arr2)))
    t_stat = (median1 - median2) / pooled_mad if pooled_mad > 0 else 0
    df = len(arr1) + len(arr2) - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Brunner-Munzel test - non-parametric test for comparing two samples
    # More robust than t-test, doesn't assume equal variances
    bm_stat, bm_p_value = stats.brunnermunzel(arr1, arr2, alternative='two-sided')
    
    # Check if there's a statistical difference
    has_statistical_difference = (p_value < 0.05) or (bm_p_value < 0.05)
    
    if has_statistical_difference:
        # Create overlapping bar charts
        plt.figure(figsize=(8, 6))
        
        # Create bins for histogram-like bar chart
        all_data = np.concatenate([arr1, arr2])
        bins = np.linspace(min(all_data), max(all_data), 20)
        
        # Create histograms
        hist1, _ = np.histogram(arr1, bins=bins)
        hist2, _ = np.histogram(arr2, bins=bins)
        
        # Calculate bin centers for bar positioning
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]
        
        # Create overlapping bar chart
        plt.bar(bin_centers - bin_width/4, hist1, width=bin_width/2, 
                alpha=0.7, label=list1_label, color='blue')
        plt.bar(bin_centers + bin_width/4, hist2, width=bin_width/2, 
                alpha=0.7, label=list2_label, color='red')
        
        # Add vertical lines for medians
        plt.axvline(median1, color='blue', linestyle='--', linewidth=2, 
                    label=f'{list1_label} Median: {median1:.3f}')
        plt.axvline(median2, color='red', linestyle='--', linewidth=2, 
                    label=f'{list2_label} Median: {median2:.3f}')
        
        # Formatting
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title(f'Overlapping Distribution Comparison\n'
                  f'MAD₁: {mad1:.3f}, MAD₂: {mad2:.3f}\n'
                  f'MAD-based p-value: {p_value:.6f}, Brunner-Munzel p-value: {bm_p_value:.6f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot and show non-blocking
        plt.tight_layout()
        plt.savefig(f'{output_folder}/comparison_plot{s_suffix}.png', dpi=300, bbox_inches='tight')
        plt.show(block=False)
        
        # Print summary statistics
        print(f"\n=== Comparison Results ===")
        print(f"{list1_label}:")
        print(f"  Median: {median1:.6f}")
        print(f"  MAD: {mad1:.6f}")
        print(f"  Count: {len(arr1)}")
        
        print(f"\n{list2_label}:")
        print(f"  Median: {median2:.6f}")
        print(f"  MAD: {mad2:.6f}")
        print(f"  Count: {len(arr2)}")
        
        print(f"\nStatistical Tests:")
        print(f"  Median Difference: {abs(median1 - median2):.6f}")
        print(f"  MAD-based t-test p-value: {p_value:.6f}")
        print(f"  MAD-based test significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
        print(f"  Brunner-Munzel statistic: {bm_stat:.6f}")
        print(f"  Brunner-Munzel p-value: {bm_p_value:.6f}")
        print(f"  Brunner-Munzel test significant at α=0.05: {'Yes' if bm_p_value < 0.05 else 'No'}")
    else:
        print("no statistical difference")
    
    return {
        'median1': median1,
        'median2': median2,
        'mad1': mad1,
        'mad2': mad2,
        'p_value': p_value,
        'median_difference': abs(median1 - median2),
        'brunner_munzel_statistic': bm_stat,
        'brunner_munzel_p_value': bm_p_value
    }

def test_compare_lists_with_plots(output_folder: str = "") -> None:
    print("\n=== Testing compare_lists_with_plots function ===")
    
    # Generate sample data for testing
    np.random.seed(42)  # For reproducible results
    list1 = np.random.normal(10, 2, 50).tolist()  # Normal distribution, mean=10, std=2
    list2 = np.random.normal(12, 3, 60).tolist()  # Normal distribution, mean=12, std=3
    
    # Test the function
    results = compare_lists_with_plots(list1, list2, "Sample A", "Sample B", output_folder=output_folder)
    print(results)

def compare_costs(s_suffix: str = "", output_folder: str = "", d_l_e_type_costs: Optional[Dict[str, List[int]]] = None) -> None:
    if d_l_e_type_costs is None:
        d_l_e_type_costs = {}
    
    comp = compare_lists_with_plots(d_l_e_type_costs["supports"], d_l_e_type_costs["contradicts"], "Supports", "Contradicts", s_suffix, output_folder)
    
    # Convert d_l_e_type_costs to dataframe and save
    max_len = max(len(d_l_e_type_costs[key]) for key in d_l_e_type_costs.keys()) if d_l_e_type_costs else 0
    costs_data = {}
    for key, values in d_l_e_type_costs.items():
        # Pad shorter lists with NaN to make all columns the same length
        padded_values = values + [np.nan] * (max_len - len(values))
        costs_data[key] = padded_values
    
    df_costs = pd.DataFrame(costs_data)
    df_costs.to_csv(f"{output_folder}/costs_data{s_suffix}.csv", index=False)
    print(f"Saved d_l_e_type_costs to {output_folder}/costs_data{s_suffix}.csv")
    
    # Convert comp results to dataframe and save
    comp_df = pd.DataFrame([comp])  # Single row with all the comparison results
    comp_df.to_csv(f"{output_folder}/comparison_results{s_suffix}.csv", index=False)
    print(f"Saved comp results to {output_folder}/comparison_results{s_suffix}.csv")

def compare_toks(s_suffix: str = "", d_h_pre: Optional[Dict[str, str]] = None, output_folder: str = "", l_d_tok: Optional[List[Dict[str, Any]]] = None) -> None:
    if d_h_pre is None:
        d_h_pre = {}
    if l_d_tok is None:
        l_d_tok = []
    
    df = pd.DataFrame(l_d_tok)
    df.to_csv(f"{output_folder}/toks_data{s_suffix}.csv", index=False)
    print(f"Saved l_d_tok to {output_folder}/toks_data{s_suffix}.csv")

    for t in g_l_token_types_to_eval:
        l_sup = list(df[(df["type"] == "e") & (df["mode"] == "supports")][t])
        l_cont = list(df[(df["type"] == "e") & (df["mode"] == "contradicts")][t])
        comp = compare_lists_with_plots(l_sup, l_cont, "Supports", "Contradicts", s_suffix + "-Evidence-gen-costs-" + t, output_folder)
        comp_df = pd.DataFrame([comp])  # Single row with all the comparison results
        comp_df.to_csv(f"{output_folder}/comparison_results{s_suffix}-Evidence-gen-costs-{t}.csv", index=False)        

    for t in g_l_token_types_to_eval:
        l_sup = list(df[(df["type"] == "r") & (df["mode"].isin(["1", "3"]))][t])
        l_cont = list(df[(df["type"] == "r") & (df["mode"].isin(["-1", "-3"]))][t])
        comp = compare_lists_with_plots(l_sup, l_cont, "Supports", "Opposes", s_suffix + "-Support-degree-costs-" + t, output_folder)
        comp_df = pd.DataFrame([comp])  # Single row with all the comparison results
        comp_df.to_csv(f"{output_folder}/comparison_results{s_suffix}-Support-degree-costs-{t}.csv", index=False)

    if (len(d_h_pre) > 0):
        l_true_events = [key for key, value in d_h_pre.items() if str(value) == "True"]
        l_false_events = [key for key, value in d_h_pre.items() if str(value) == "False"]

        df_true = df[df["event"].str.split(magic_split).str[0].isin(l_true_events)]
        for t in g_l_token_types_to_eval:
            l_sup = list(df_true[(df_true["type"] == "e") & (df_true["mode"] == "supports")][t])
            l_cont = list(df_true[(df_true["type"] == "e") & (df_true["mode"] == "contradicts")][t])
            comp = compare_lists_with_plots(l_sup, l_cont, "Supports", "Contradicts", s_suffix + "-Evidence-gen-costs-for-true-events-" + t, output_folder)
            comp_df = pd.DataFrame([comp])  # Single row with all the comparison results
            comp_df.to_csv(f"{output_folder}/comparison_results{s_suffix}-Evidence-gen-costs-for-true-events-{t}.csv", index=False)

        for t in g_l_token_types_to_eval:
            l_sup = list(df_true[(df_true["type"] == "r") & (df_true["mode"].isin(["1", "3"]))][t])
            l_cont = list(df_true[(df_true["type"] == "r") & (df_true["mode"].isin(["-1", "-3"]))][t])
            comp = compare_lists_with_plots(l_sup, l_cont, "Supports", "Opposes", s_suffix + "-Support-degree-costs-for-true-events-" + t, output_folder)
            comp_df = pd.DataFrame([comp])  # Single row with all the comparison results
            comp_df.to_csv(f"{output_folder}/comparison_results{s_suffix}-Support-degree-costs-for-true-events-{t}.csv", index=False)            

        df_false = df[df["event"].str.split(magic_split).str[0].isin(l_false_events)]
        for t in g_l_token_types_to_eval:
            l_sup = list(df_false[(df_false["type"] == "e") & (df_false["mode"] == "supports")][t])
            l_cont = list(df_false[(df_false["type"] == "e") & (df_false["mode"] == "contradicts")][t])
            comp = compare_lists_with_plots(l_sup, l_cont, "Supports", "Contradicts", s_suffix + "-Evidence-gen-costs-for-false-events-" + t, output_folder)
            comp_df = pd.DataFrame([comp])  # Single row with all the comparison results
            comp_df.to_csv(f"{output_folder}/comparison_results{s_suffix}-Evidence-gen-costs-for-false-events-{t}.csv", index=False)

        for t in g_l_token_types_to_eval:
            l_sup = list(df_false[(df_false["type"] == "r") & (df_false["mode"].isin(["1", "3"]))][t])
            l_cont = list(df_false[(df_false["type"] == "r") & (df_false["mode"].isin(["-1", "-3"]))][t])
            comp = compare_lists_with_plots(l_sup, l_cont, "Supports", "Opposes", s_suffix + "-Support-degree-costs-for-false-events-" + t, output_folder)
            comp_df = pd.DataFrame([comp])  # Single row with all the comparison results
            comp_df.to_csv(f"{output_folder}/comparison_results{s_suffix}-Support-degree-costs-for-false-events-{t}.csv", index=False)                          

# A fully AI-written method. You can see its thinking style.
# All correct, easy to maintain... but no human would've written it like this :)
def estimate_get_matrix_cost(d_config: Dict[str, Any]) -> float:
    """
    Estimates the dollar cost of calling get_matrix method with the given parameters.
    
    Parameters:
    - d_config: Configuration dictionary containing all parameters
    
    Returns:
    - Estimated cost in dollars
    """
    # Extract parameters from config
    nH = d_config["nH"]
    nE_per_H = d_config["nE"]
    nMaxHLen = d_config.get("nMaxHLen", 128)
    nMaxELen = d_config.get("nMaxELen", 300)
    d_h_pre = d_config["ExtraH"]
    l_h_pre = list(d_h_pre.keys())
    l_e_pre = d_config["ExtraE"]
    model_in = d_config["model"]
    b_fill_matrix = d_config.get("b_fill_matrix", True)
    
    # Base token estimates for first call
    base_input_tokens = 200
    base_reasoning_tokens = 2000
    
    # Estimate tokens per character (rough approximation: 4 chars per token)
    tokens_per_hypothesis = 1.5*nMaxHLen
    tokens_per_evidence = 2*nMaxELen
    
    # Get cost per million tokens for the model
    cost_per_M = d_API_cost_per_M[model_in]  # Lets explode if the model is unknown
    
    total_input_tokens = 0
    total_reasoning_tokens = 0
    total_output_tokens = 0
    
    # 1. Estimate ask_h calls
    num_h_pre = len(l_h_pre)
    for i in range(nH):
        # Input tokens: base + previous hypotheses
        # Each previous hypothesis adds tokens_per_hypothesis tokens
        prev_h_count = num_h_pre + i
        input_tokens = base_input_tokens + (prev_h_count * tokens_per_hypothesis)
        reasoning_tokens = base_reasoning_tokens
        output_tokens = tokens_per_hypothesis
        
        total_input_tokens += input_tokens
        total_reasoning_tokens += reasoning_tokens
        total_output_tokens += output_tokens
    
    # 2. Estimate ask_e calls
    total_hypotheses = num_h_pre + nH
    num_e_pre = len(l_e_pre)
    
    # For each hypothesis, we generate nE_per_H supports and nE_per_H contradicts
    evidence_count = 0
    for h_idx in range(total_hypotheses):
        for e_type in range(nE_per_H * 2):  # supports and contradicts
            # Input tokens: base + previous evidence
            # Each previous evidence adds tokens_per_evidence tokens
            prev_e_count = num_e_pre + evidence_count
            input_tokens = base_input_tokens + (prev_e_count * tokens_per_evidence)
            reasoning_tokens = base_reasoning_tokens
            output_tokens = tokens_per_evidence
            
            total_input_tokens += input_tokens
            total_reasoning_tokens += reasoning_tokens
            total_output_tokens += output_tokens
            
            evidence_count += 1
    
    # 3. Estimate cross_ref calls (if b_fill_matrix is True)
    if b_fill_matrix:
        total_evidence = num_e_pre + (total_hypotheses * nE_per_H * 2)
        num_cross_refs = total_hypotheses * total_evidence
        
        # cross_ref calls are simpler, don't accumulate context
        # Estimate smaller base for cross_ref
        cross_ref_base_input = (nMaxHLen + nMaxELen)/2
        cross_ref_reasoning = 1200
        cross_ref_output = 2  # Just a number
        
        for _ in range(num_cross_refs):
            total_input_tokens += cross_ref_base_input
            total_reasoning_tokens += cross_ref_reasoning
            total_output_tokens += cross_ref_output
    
    # Calculate total cost
    # Cost = (input_tokens + output_tokens + reasoning_tokens) * cost_per_M / 1,000,000
    total_tokens = total_input_tokens + total_output_tokens + total_reasoning_tokens
    estimated_cost = (total_tokens * cost_per_M) / 1e6
    
    return estimated_cost


