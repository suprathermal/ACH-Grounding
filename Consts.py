# Constants used across the application

# Magic split delimiter for event strings
magic_split = " @vs@ "

# Supported support degree values
g_l_supported_values = [3, 1, 0, -1, -3]
# Tuple version for use in Literal type annotations (sorted for consistency)
g_t_supported_values = tuple(sorted(g_l_supported_values))

d_API_cost_per_M = {"gpt-5-mini": 0.250, 
                    "gpt-5-nano": 0.050,
                    "gpt-5":1.250,
                    "gpt-5-1":1.250,
                    "gpt-5-pro":15.00
                    }


