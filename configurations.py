
# TODO: Adjust to my environment
four_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3}
eight_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3, 'WTWL': 4, 'ETEL': 5, 'STSL': 6, 'NTNL': 7}
location_dict = {"N": "North", "S": "South", "E": "East", "W": "West"}
location_dict_detail = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}
direction_dict = {"T": "through", "L": "left-turn", "R": "turn-right"}
direction_dict_ori = {"T": "through", "L": "turn-left", "R": "turn-right"}

phase_explanation_dict_detail = {"NTST": "- NTST: Northern and southern through lanes.",
                                 "NLSL": "- NLSL: Northern and southern left-turn lanes.",
                                 "NTNL": "- NTNL: Northern through and left-turn lanes.",
                                 "STSL": "- STSL: Southern through and left-turn lanes.",
                                 "ETWT": "- ETWT: Eastern and western through lanes.",
                                 "ELWL": "- ELWL: Eastern and western left-turn lanes.",
                                 "ETEL": "- ETEL: Eastern through and left-turn lanes.",
                                 "WTWL": "- WTWL: Western through and left-turn lanes."
                                }

incoming_lane_2_outgoing_road = {
    "NT": "South",
    "NL": "East",
    "ST": "North",
    "SL": "West",
    "ET": "West",
    "EL": "South",
    "WT": "East",
    "WL": "North"
}


# only works with 3 lanedroads
dataset_phase_configs = {
"ETWT_green": "gggrrrrrrgggGGGrrrgggrrrrrrgggGGGrrr",
"ETWT_yellow": "gggrrrrrrgggyyyrrrgggrrrrrrgggyyyrrr",
"ELWL_green": "gggrrrrrrgggrrrGGGgggrrrrrrgggrrrGGG",
"ELWL_yellow": "gggrrrrrrgggrrryyygggrrrrrrgggrrryyy",
"NTST_green": "gggGGGrrrgggrrrrrrgggGGGrrrgggrrrrrr",
"NTST_yellow": "gggyyyrrrgggrrrrrrgggyyyrrrgggrrrrrr",
"NLSL_green": "gggrrrGGGgggrrrrrrgggrrrGGGgggrrrrrr",
"NLSL_yellow": "gggrrryyygggrrrrrrgggrrryyygggrrrrrr",
"all_red": "gggrrrrrrgggrrrrrrgggrrrrrrgggrrrrrr"

}