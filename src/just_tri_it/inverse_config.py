from just_tri_it.utils import config_match
from typing import Any

CONFIG_MAP = {
    "keep_length": [{
        "problems": [
            "11_binary_string", "atcoder_abc393_d",
            "atcoder_abc395_e", "atcoder_abc396_c", 
            "atcoder_abc391_e", "atcoder_abc391_g",
            "atcoder_abc394_f", "atcoder_abc396_a",
            "atcoder_abc398_c", "atcoder_abc397_c",
            "atcoder_abc390_b", "atcoder_abc399_b",
            "atcoder_abc399_f", "leetcode_3764"
        ],
        "model": None,
        "msg": None
    },
    {   
        "problems": [
            "atcoder_abc390_d", "leetcode_3781",
            "atcoder_abc398_g", "atcoder_arc192_a",
            "atcoder_abc399_d", "atcoder_abc397_g",
            "atcoder_arc191_d", "atcoder_abc396_e",
            "atcoder_abc398_d", "atcoder_abc388_d"
        ], 
        "model": ["deepseek-v3"],
        "msg": None
    }],
    "add_length": [{
        "problems": ["2_list_sum"],
        "model": None,
        "msg": None
    }],
    "remove_length": [{
        "problems": ["choose_your_queries"],
        "model": None,
        "msg": (0, 2)
    },
    {
        "problems": [
            "and_reconstruction", "concatenation_of_arrays",
            "earning_on_bets", "grid_reset", "manhattan_triangle",
            "slavics_exam", "stardew_valley", "xorificator",
            "find_k_distinct_points", "strong_password"
        ],
        "model": None,
        "msg": (0, 1)
    }],
    "remove_add_length": [{
        "problems": ["atcoder_abc388_c"],
        "model": ["gpt-4o"],
        "msg": None
    }],
    "stream_process": [{
        "problems": [
            "atcoder_arc191_c", "and_reconstruction", 
            "concatenation_of_arrays", "earning_on_bets",
            "grid_reset", "manhattan_triangle",
            "slavics_exam", "common_generator",
            "cool_graph", "stardew_valley", "xorificator",
            "find_k_distinct_points", "strong_password"
        ],
        "model": None,
        "msg": None
    }],
    "pattern_spec": [
    # round 5: re-invert remaining false-selection tasks to whole structured
    # parameters (or a different argument) so that no false agreement forms
    # (gemini-2.5-flash only; indices are post-length-removal)
    {
        # arc192_b: whole position_values; arc194_a: whole integer_sequence;
        # leetcode_3762: whole points
        "problems": [
            "atcoder_arc192_b", "atcoder_arc194_a", "leetcode_3762"
        ],
        "model": ["gemini-2.5-flash"],
        "msg": 0
    },
    {
        # abc389_e: whole product_prices (post-simp: max_cost, product_prices)
        "problems": ["atcoder_abc389_e"],
        "model": ["gemini-2.5-flash"],
        "msg": 1
    },
    {
        # abc397_g: whole edges (post-simp: N, K, edges)
        "problems": ["atcoder_abc397_g"],
        "model": ["gemini-2.5-flash"],
        "msg": 2
    },
    # lcb_improvement.md re-inversions (gemini-2.5-flash only); indices are
    # relative to the post-length-removal signature
    {
        # abc388_b: length_increase; abc393_e: elements_to_choose;
        # abc398_g: num_vertices; leetcode_3697: nums;
        # abc388_d: whole initial_stones (single param after simp);
        # abc396_d: num_vertices (post-simp: num_vertices, edges)
        "problems": [
            "atcoder_abc388_b", "atcoder_abc393_e",
            "atcoder_abc398_g",
            "leetcode_3697", "atcoder_abc388_d",
            "atcoder_abc396_d"
        ],
        "model": ["gemini-2.5-flash"],
        "msg": 0
    },
    {
        # abc393_f: suffix (last element) of sequence
        "problems": ["atcoder_abc393_f"],
        "model": ["gemini-2.5-flash"],
        "msg": (0, 1, "list")
    },
    {
        # abc396_g: suffix (last row) of grid (post-simp: W, grid)
        "problems": ["atcoder_abc396_g"],
        "model": ["gemini-2.5-flash"],
        "msg": (1, 1, "list")
    },
    {
        # arc194_c: costs
        "problems": ["atcoder_arc194_c"],
        "model": ["gemini-2.5-flash"],
        "msg": 2
    },
    {
        # arc196_d: query_ranges (post-simp: num_towns, num_queries,
        # start_towns, end_towns, query_ranges)
        "problems": ["atcoder_arc196_d"],
        "model": ["gemini-2.5-flash"],
        "msg": 4
    },
    {
        "problems": [
            "11_binary_string", "atcoder_abc393_d",
            "atcoder_abc391_e", "turtle_and_good_pairs"
        ],
        "model": None,
        "msg": (1, 1, "str")
    },
    {
        "problems": [
            "2_list_sum", "atcoder_abc399_b",
            "atcoder_abc394_f", "absolute_zero",
            "alices_adventures_in_cards", "choose_your_queries",
            "common_generator", "concatenation_of_arrays",
            "turtle_incomplete_sequence",
            "and_reconstruction", "atcoder_abc396_a",
            "atcoder_abc398_c", "atcoder_abc390_b",
        ],
        "model": None,
        "msg":(1, 1, "list")
    },
    {
        "problems": ["leetcode_3720", "leetcode_3789"],
        "model": ["deepseek-v3"],
        "msg": (1, 1, "list")
    },
    {
        "problems": ["atcoder_abc388_c"],
        "model": ["gpt-4o"],
        "msg": (1, 1, "list")
    },
    {
        "problems": [
            "atcoder_abc388_c", "leetcode_3759",
            "leetcode_3714", "leetcode_3751",
            "atcoder_arc194_b", "leetcode_3765"
        ],
        "model": ["deepseek-v3"],
        "msg": (0, 3, "list")
    },
    {
        "problems": ["leetcode_3785"],
        "model": None,
        "msg": (0, 1, "list")
    },
    {
        "problems": ["atcoder_abc390_a", "atcoder_arc195_a"],
        "model": None,
        "msg": (0, 2, "list")
    },
    {
        "problems": ["leetcode_3722"],
        "model": ["deepseek-v3"],
        "msg": (0, 2, "list")
    },
    {
        "problems": ["atcoder_abc395_e"],
        "model": None,
        "msg": (3, 1, "list")
    },
    {
        "problems": ["atcoder_abc396_c", "cool_graph", "stardew_valley", "atcoder_abc399_f"],
        "model": None,
        "msg": (2, 1, "list")
    },
    {
        "problems": ["atcoder_abc391_g"],
        "model": ["gpt-4o"],
        "msg": (2, 1, "str")
    },
    {
        "problems": [
            "atcoder_arc196_a", "leetcode_3754",
            "leetcode_3771", "leetcode_3717",
            "atcoder_arc191_a", "leetcode_3788",
            "atcoder_abc395_a"
        ],
        "model": ["deepseek-v3"],
        "msg": 0
    },
    {
        "problems": ["leetcode_3832", "leetcode_3770", "perpendicular_segments"],
        "model": None,
        "msg": 0
    },
    {
        "problems": ["atcoder_abc394_f", "atcoder_abc399_d", "atcoder_arc192_a", "atcoder_abc387_f"],
        "model": ["deepseek-v3"],
        "msg": (1, 3, "list")
    },
    {
        "problems": ["atcoder_abc390_d", "leetcode_3781", "atcoder_abc399_e"],
        "model": ["deepseek-v3"],
        "msg": 1
    },
    {
        "problems": ["manhattan_triangle", "slavics_exam", "atcoder_abc400_a", "atcoder_abc400_b"],
        "model": None,
        "msg": 1
    },
    {
        "problems": ["atcoder_abc397_g"],
        "model": ["deepseek-v3"],
        "msg": (3, 3, "list")
    },
    {
        "problems": ["strong_password", "leetcode_3793", "leetcode_3709"],
        "model": None,
        "msg": (0, 1, "str")
    },
    {
        "problems": ["atcoder_abc388_d"],
        "model": ["deepseek-v3"],
        "msg": (1, 2, "list")
    },
    {
        "problems": ["and_reconstruction"],
        "model": None,
        "msg": (1, 2, "list")
    },
    {
        "problems": ["earning_on_bets"],
        "model": None,
        "msg": (1, 2, "list")
    },
    {
        "problems": ["atcoder_abc396_e", "atcoder_arc194_c", "atcoder_abc398_g"],
        "model": ["deepseek-v3"],
        "msg": 2
    },
    {
        "problems": ["atcoder_abc398_d"],
        "model": ["deepseek-v3"],
        "msg": 3
    },
    {
        "problems": ["atcoder_arc191_d"],
        "model": ["deepseek-v3"],
        "msg": 4
    },
    {
        "problems": ["atcoder_abc391_g"],
        "model": ["deepseek-v3"],
        "msg": (2, 3, "str")
    },
    {
        "problems": ["atcoder_abc391_a"],
        "model": ["gpt-4o"],
        "msg": (0, 3, "str")
    },
    {
        "problems": ["ingenuity_2"],
        "model": None,
        "msg": (1, 2, "str")
    },
    {
        "problems": ["xorificator"],
        "model": None,
        "msg": (2, 2, "list")
    }],
    "yes_no": [{
        "problems": ["slavics_exam"],
        "model": None,
        "msg": None
    }],
    "name_spec": [{
        "problems": ["slavics_exam"],
        "model": None,
        "msg": "s_with_replaced_marks"
    }],
    "des_spec": [{
        "problems": ["manhattan_triangle"],
        "model": None,
        "msg": ("a tuple of three distinct integers representing indices in", "a tuple of three distinct integers representing indices (from 1 to n inclusive) in")
    }],
    "only_timeout": [{
        "problems": [
            "atcoder_abc388_c", "atcoder_abc391_f",
            "atcoder_abc397_f", "leetcode_3722",
            "leetcode_3720", "leetcode_3714",
            "leetcode_3717", "atcoder_abc388_d",
            "atcoder_abc400_e", "atcoder_abc388_f",
            "atcoder_abc387_c", "atcoder_abc392_g",
            "leetcode_3674", "leetcode_3725"
        ],
        "model": None,
        "msg": None
    }],
    "return_spec": [
    {
        "problems": ["slavics_exam"],
        "model": None,
        "msg": "no_or_yes_and_replaced_string"
    },
    {
        "problems": ["atcoder_abc393_e"],
        "model": None,
        "msg": "maximum_gcd_list"
    }],
    "bound_spec": [{
        "problems": ["slavics_exam"],
        "model": None,
        "msg": 3
    },
    {
        # abc391_g: keep generated inputs small (string_s of length <= 3) so
        # exhaustive sinv enumerators finish within the execution timeout
        "problems": ["atcoder_abc391_g"],
        "model": ["gemini-2.5-flash"],
        "msg": 3
    }],
    "validate_length": [{
        # leetcode_3709: reject the length-parameter claim unless it holds on
        # all generated inputs (k is a substring-length requirement, not len(s))
        "problems": ["leetcode_3709"],
        "model": ["gemini-2.5-flash"],
        "msg": None
    }],
    "sinv_extra_spec": [{
        # leetcode_3770: the forward output may be an impossible-case sentinel;
        # instruct the inverse to reject it as invalid input instead of crashing
        "problems": ["leetcode_3770"],
        "model": ["gemini-2.5-flash"],
        "msg": ("Additional requirement: if the given desired output value is "
                "an empty string (which indicates that generating a valid "
                "result was impossible), the function must raise "
                "ValueError('Invalid input') instead of returning a value.")
    }],
    "sinv_tolerate_invalid": [{
        # leetcode_3770: map ValueError('Invalid input') raised by the inverse
        # to a tolerated (Angelic) outcome instead of a hard failure
        "problems": ["leetcode_3770"],
        "model": ["gemini-2.5-flash"],
        "msg": None
    }],
    "enable_filter": [{
        "problems": ["leetcode_3759", "leetcode_3793"],
        "model": None,
        "msg": None
    }],
    "in_spec": [{
        "problems": ["slavics_exam"],
        "model": None,
        "msg": ['ba', 'a?']
    }],
    "member_spec": [{
        "problems": ["manhattan_triangle"],
        "model": None,
        "msg": None
    }],
    "small_filter": [{
        "problems": ["slavics_exam"],
        "model": None,
        "msg": None
    }],
}

def config(config_type: str) -> tuple[bool, Any]:
    try:
        config_list = CONFIG_MAP[config_type]
    except KeyError:
        print(f"don't have the key {config_type}")
        return (False, None)

    for config in config_list:
        if config_match(task=config["problems"], model=config["model"]):
            return (True, config["msg"])
    return (False, None)
