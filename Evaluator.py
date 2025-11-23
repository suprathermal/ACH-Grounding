import numpy as np
import pandas as pd
from typing import List
from Consts import g_l_supported_values
"""
Evaluates the support matrix via one of several methods
"""

def get_std_with_boundary_uncertainty(lIn: List[int]) -> float:
    l_seen_values = list(set(lIn))
    l_unseen_values = [x for x in g_l_supported_values if x not in l_seen_values]
    if (0 == len(l_unseen_values)):
        return np.std(lIn)
    
    # Find the element with the maximum absolute value (first if multiple exist)
    max_abs_value = max(abs(x) for x in l_unseen_values)
    l_matches = [x for x in l_unseen_values if abs(x) == max_abs_value]
    max_abs_element = l_matches[0] # Yes, we intentionally assume there is a match; any other outcome would mean gliding over some weird bug
    l_seen_values.append(max_abs_element)
    if (1 == len(l_seen_values)):
        return np.std(g_l_supported_values)
    else:
        return np.std(l_seen_values)
    
# Per https://en.wikipedia.org/wiki/Analysis_of_competing_hypotheses
def eval_ACH_basic(dfIn: pd.DataFrame) -> pd.DataFrame:
    df = dfIn.copy()

    l_d_ret = []
    # Replace any values not in g_l_supported_values with zeros in all columns except "E"
    for col in df.columns:
        if col != "E":
            df[col] = df[col].apply(lambda x: 0 if x not in g_l_supported_values else x)

    l_h = [c for c in df.columns if c != "E"]
    for h in l_h:
        n = len(l_h)
        avg = df[h].mean()
        l_values = df[h].values.tolist()
        std = get_std_with_boundary_uncertainty(l_values)
        err = std / np.sqrt(n)
        l_d_ret.append({"hypothesis": h, "avg": avg, "stderr": err, "support":n})

    return pd.DataFrame(l_d_ret)

# Classic ACH suffers from several serious shortcomings:
# 1. It is unstable to duplicative evidence. Multiple repetitions of the same or highly similar evidence could result in an arbitrarily controlled output.
# 2. It is unstable to noise. Large number of completely random evidence rows could result in (eventually) any outcome, statsig
# 3. It produces scores, whereas we care about probabilities. 
# 4. It does not warn if a conclusion is based on just 1-2 strong but possibly spurious observations
# The method below addresses all of these concerns.
def eval_ACH(dfIn: pd.DataFrame) -> pd.DataFrame:
    raise Exception("Implementation not provided. Please contact the author for details.")

# Either classic or advanced ACH is oblivious to the truthfulness of the hypotheses.
# Yet, for some of them that information may be reliably available.
# This approach takes that into account to derive the truthfulness of all hypotheses
def eval_ML(dfIn: pd.DataFrame) -> pd.DataFrame:
    raise Exception("Implementation not provided. Please contact the author for details.")

# Same as above, but stable to some incorrect labels on the hypotheses, or
# to truthfulness driven by 2+ competing models
def eval_exhaustive(dfIn: pd.DataFrame) -> pd.DataFrame:
    raise Exception("Implementation in progress. Please contact the author for details.")