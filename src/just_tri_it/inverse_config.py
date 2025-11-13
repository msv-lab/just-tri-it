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
            "atcoder_abc399_f"
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
    "pattern_spec": [{
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
            "earning_on_bets", "turtle_incomplete_sequence",
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
