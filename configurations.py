# # configurations.py

location_dict = {"N": "North", "S": "South", "E": "East", "W": "West"}
location_dict_detail = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}

incoming_lane_2_outgoing_road = {
    "NT": "South", "NL": "East",
    "ST": "North", "SL": "West",
    "ET": "West",  "EL": "South",
    "WT": "East",  "WL": "North"
}

# The centralized configuration dictionary
INTERSECTION_CONFIG = {
    "global_settings": {
        "all_red_state": "gggrrrrrrgggrrrrrrgggrrrrrrgggrrrrrr",
        "default_green_duration": 30,
        "yellow_duration": 3,
        "red_duration": 2
    },
    "phases": {
        "ETWT": {
            "id": 0,
            "green": "gggrrrrrrgggGGGrrrgggrrrrrrgggGGGrrr",
            "yellow": "gggrrrrrrgggyyyrrrgggrrrrrrgggyyyrrr",
            "llm_description": "- ETWT: Eastern and western through lanes."
        },
        "NTST": {
            "id": 1,
            "green": "gggGGGrrrgggrrrrrrgggGGGrrrgggrrrrrr",
            "yellow": "gggyyyrrrgggrrrrrrgggyyyrrrgggrrrrrr",
            "llm_description": "- NTST: Northern and southern through lanes."
        },
        "ELWL": {
            "id": 2,
            "green": "gggrrrrrrgggrrrGGGgggrrrrrrgggrrrGGG",
            "yellow": "gggrrrrrrgggrrryyygggrrrrrrgggrrryyy",
            "llm_description": "- ELWL: Eastern and western left-turn lanes."
        },
        "NLSL": {
            "id": 3,
            "green": "gggrrrGGGgggrrrrrrgggrrrGGGgggrrrrrr",
            "yellow": "gggrrryyygggrrrrrrgggrrryyygggrrrrrr",
            "llm_description": "- NLSL: Northern and southern left-turn lanes."
        }
    }
}

