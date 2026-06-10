from configurations import INTERSECTION_CONFIG

def state_dict_to_csv(state_dict):
    """
    Converts the state dictionary from SumoEnv into the CSV text format 
    expected by the prompt builder.
    """
    # Header required by the prompt's instructions
    header = "Lane,Early_Queued,Avg_Wait_Time,Segment_1,Segment_2,Segment_3,Segment_4"
    rows = [header]
    
    # Extract lane states from the dictionary
    lane_states = state_dict.get("lane_states", {})
    
    for lane_id, data in lane_states.items():
        queued = data.get("early_queued", 0)
        
        # Defaulting wait time to 0.0 since it is not currently tracked in SumoEnv
        wait_time = 0.0 
        
        segments = data.get("segments", {})
        seg1 = segments.get("segment_1", 0)
        seg2 = segments.get("segment_2", 0)
        seg3 = segments.get("segment_3", 0)
        
        # Defaulting segment 4 to 0 since SumoEnv only calculates 3 segments
        seg4 = 0 
        
        # Build the CSV row for this lane
        row = f"{lane_id},{queued},{wait_time},{seg1},{seg2},{seg3},{seg4}"
        rows.append(row)
        
    return "\n".join(rows)

def getPrompt(state_dict, avg_speed, system_prompt, length_dict, phases):
    state_txt = state_dict_to_csv(state_dict)
    # fill information
    signals_text = ""
    for p in phases:
        # Access the llm_description from the new centralized configuration
        if p in INTERSECTION_CONFIG["phases"]:
            signals_text += INTERSECTION_CONFIG["phases"][p]["llm_description"] + "\n"
        else:
            signals_text += f"- {p}: Unknown phase description.\n"

    prompt = [
        {"role": "system",
         "content": system_prompt},
        {"role": "user",
         "content": "A traffic light regulates a four-section intersection with northern, southern, eastern, and "
                    "western sections, each containing two lanes: one for through traffic and one for left-turns. "
                    f"The eastern and western lanes are {int(length_dict['East'])} meters long, while the northern and southern lanes are "
                    f"{int(length_dict['North'])} meters in length. Each lane is further divided into four segments. Segment 1 spans from the "
                    "10m mark of the lane to segment 2. Segment 2 begins at the 1/10 mark of the lane and links segment "
                    "1 to segment 3. Segment 3 starts at the 1/3 mark of the lane and links segment 2 to segment 4. "
                    "Segment 4 begins at the 2/3 mark of the lane, spanning from the end of segment 3 to the lane's end.\n\n"
                    "The current lane statuses are:\n" + state_txt + "\n" +
                    "This CSV table shows lane statuses, with the first column representing lanes, the second column "
                    "displaying early queued vehicle counts, the third column showing the average time that early "
                    "queued vehicles have waited in previous phases, and columns 4-7 indicating approaching vehicle "
                    "counts in the four lane segments.\n\n"
                    "Early queued vehicles have arrived at the intersection and await passage permission. Approaching "
                    f"vehicles are at an average speed of {int(avg_speed)}m/s. If they can arrive at the intersection during the next "
                    "phase, they may merge into the appropriate waiting queues (if they are NOT allowed to pass) or "
                    "pass the intersection (if they are allowed to pass).\n\n"
                    f"The traffic light has {len(phases)} signal phases. Each signal relieves vehicles' flow in the two specific "
                    "lanes. The lanes relieving vehicles' flow under each traffic light phase are listed below:\n" +
                    signals_text +
                    "\nThe next signal phase will persist for 30 seconds.\n\n"
                    "Please follow the following steps to provide your analysis (pay attention to accurate variable calculations in each step):\n"
                    "- Step 1: Calculate the ranges of the four lane segments in different lanes.\n"
                    "- Step 2: Identify the lane segments that vehicles travel on can potentially reach the intersection within the next phase.\n"
                    "- Step 3: Analyzing the CSV table, identify the traffic conditions (early queued vehicle count, average waiting time, and the approaching vehicle count in segments identified in Step 2) in each lane.\n"
                    "- Step 4: If no vehicle is permitted to pass the intersection within the next phase, analyze:\n"
                    "a) The total cumulative waiting times of ALL early queued vehicles that will accumulate by the END of the next phase in each lane.\n"
                    "b) The total waiting times of ALL vehicles from reachable segments within the next phase in each lane.\n"
                    "c) The total waiting time of ALL queuing vehicles analyzed above in each lane.\n"
                    "- Step 5: Considering the total waiting time, analyze the potential congestion level of the two allowed lanes of each signal if vehicles on these lanes cannot be relieved in the next phase.\n"
                    "- Step 6: Considering the potential congestion level of the two allowed lanes of each signal, identify the most effective traffic signal that will most significantly improve the traffic condition during the next phase, which relieves vehicles' flow of the allowed lanes of the signal.\n\n"
                    "Requirements:\n"
                    "- Let's think step by step.\n"
                    "- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>."
         }
    ]

    return prompt