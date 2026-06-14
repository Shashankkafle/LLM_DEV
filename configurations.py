# # # configurations.py 3 lane roads

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
        },
    },
     "movement_directions": {
        "NT": "North", "NL": "North",
        "ST": "South", "SL": "South",
        "ET": "East",  "EL": "East",
        "WT": "West",  "WL": "West",
    },

    "movement_types": {
        "NT": "through", "ST": "through", "ET": "through", "WT": "through",
        "NL": "left-turn", "SL": "left-turn", "EL": "left-turn", "WL": "left-turn",
    },

    "movement_outgoing_road": {
        "NT": "South", "NL": "East",
        "ST": "North", "SL": "West",
        "ET": "West",  "EL": "South",
        "WT": "East",  "WL": "North",
    },
}




# configurations.py
# Generated for: hangzhou_1x1_bc-tyc_18041608_1h (roadnet_net.xml)
# Intersection: intersection_1_1  (single TL at x=300, y=300)
#
# === Network geometry ===
# Compass orientation (SUMO coords: x→East, y→North):
#   intersection_0_1 @ (0,   300) = West  boundary
#   intersection_2_1 @ (600, 300) = East  boundary
#   intersection_1_0 @ (300,   0) = South boundary
#   intersection_1_2 @ (300, 600) = North boundary
#
# === Incoming edges & lane IDs ===
#   North (from intersection_1_2):  road_1_2_3   lane 0 = NT,  lane 1 = NL
#   East  (from intersection_2_1):  road_2_1_2   lane 0 = ET,  lane 1 = EL
#   South (from intersection_1_0):  road_1_0_1   lane 0 = ST,  lane 1 = SL
#   West  (from intersection_0_1):  road_0_1_0   lane 0 = WT,  lane 1 = WL
#
# === Signal index mapping (16 indices) ===
#   0,1  -> NT  (North Through)
#   2,3  -> NL  (North Left-turn)
#   4,5  -> ET  (East  Through)
#   6,7  -> EL  (East  Left-turn)
#   8,9  -> ST  (South Through)
#  10,11 -> SL  (South Left-turn)
#  12,13 -> WT  (West  Through)
# #  14,15 -> WL  (West  Left-turn)

# location_dict = {"N": "North", "S": "South", "E": "East", "W": "West"}
# location_dict_detail = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}

# # Maps a movement code to the road that vehicles are heading toward
# incoming_lane_2_outgoing_road = {
#     "NT": "South", "NL": "East",
#     "ST": "North", "SL": "West",
#     "ET": "West",  "EL": "South",
#     "WT": "East",  "WL": "North"
# }





# # The centralized configuration dictionary
# INTERSECTION_CONFIG = {
#     "global_settings": {
#         "intersection_id":    "intersection_1_1",
#         "all_red_state":      "rrrrrrrrrrrrrrrr",
#         "default_green_duration": 30,
#         "yellow_duration":     3,
#         "red_duration":        2,
#     },

#     "phases": {
#         "NTST": {
#             "id": 0,
#             "green":  "GGrrrrrrGGrrrrrr",
#             "yellow": "yyrrrrrryyrrrrrr",
#             "llm_description": "- NTST: Northern and southern through lanes.",
#         },
#         "ETWT": {
#             "id": 1,
#             "green":  "rrrrGGrrrrrrGGrr",
#             "yellow": "rrrryyrrrrrryyrr",
#             "llm_description": "- ETWT: Eastern and western through lanes.",
#         },
#         "NLSL": {
#             "id": 2,
#             "green":  "rrGGrrrrrrGGrrrr",
#             "yellow": "rryyrrrrrryyrrrr",
#             "llm_description": "- NLSL: Northern and southern left-turn lanes.",
#         },
#         "ELWL": {
#             "id": 3,
#             "green":  "rrrrrrGGrrrrrrGG",
#             "yellow": "rrrrrryyrrrrrryy",
#             "llm_description": "- ELWL: Eastern and western left-turn lanes.",
#         },
#     },

#         "movement_directions": {
#         "NT": "North", "NL": "North",
#         "ST": "South", "SL": "South",
#         "ET": "East",  "EL": "East",
#         "WT": "West",  "WL": "West",
#     },

#     "movement_types": {
#         "NT": "through", "ST": "through", "ET": "through", "WT": "through",
#         "NL": "left-turn", "SL": "left-turn", "EL": "left-turn", "WL": "left-turn",
#     },

#     "movement_outgoing_road": {
#         "NT": "South", "NL": "East",
#         "ST": "North", "SL": "West",
#         "ET": "West",  "EL": "South",
#         "WT": "East",  "WL": "North",
#     },

# }