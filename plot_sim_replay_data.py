import json
import pandas as pd
import plotly.express as px

def load_and_compress_traffic_logs(filepath_or_json_str, is_string=False):
    """
    Loads traffic logs from a JSON source, filters metadata, and compresses
    consecutive identical phases per intersection into continuous timeline blocks.
    """
    # 1. Load the data
    if is_string:
        data = json.loads(filepath_or_json_str)
    else:
        with open(filepath_or_json_str, 'r') as f:
            data = json.load(f)
            
    raw_records = []
    
    # 2. Extract and parse records, discarding metadata like 'original_run_details'
    for ts_str, content in data.items():
        if ts_str == "original_run_details" or not ts_str.isdigit():
            continue
        
        timestep = int(ts_str)
        
        # Handle cases where a timestep contains a single dictionary or a list of records
        if isinstance(content, dict):
            content['timestep'] = timestep
            raw_records.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item['timestep'] = timestep
                    raw_records.append(item)

    if not raw_records:
        raise ValueError("No valid traffic log records found in the provided JSON.")

    df_raw = pd.DataFrame(raw_records)
    
    # 3. Process logs per intersection to find phase durations
    compressed_records = []
    
    for intersection_id, group in df_raw.groupby('intersection_id'):
        # Ensure chronological order
        group = group.sort_values('timestep').reset_index(drop=True)
        
        # Determine when a state ends by looking at the next logged timestep
        group['next_timestep'] = group['timestep'].shift(-1)
        
        # For the final logged record, extrapolate duration using the median step size
        median_step = group['timestep'].diff().median()
        if pd.isna(median_step) or median_step == 0:
            median_step = 1
        group['next_timestep'] = group['next_timestep'].fillna(group['timestep'] + median_step)
        
        # Initialize the first state block
        current_block = {
            'intersection_id': intersection_id,
            'phase_name': group.loc[0, 'phase_name'],
            'phase': group.loc[0, 'phase'],
            'start': int(group.loc[0, 'timestep']),
            'end': int(group.loc[0, 'next_timestep'])
        }
        
        # Compress continuous identical phases (Run-Length Encoding)
        for i in range(1, len(group)):
            row = group.iloc[i]
            if row['phase_name'] == current_block['phase_name'] and row['phase'] == current_block['phase']:
                # Extend the end boundary of the current block
                current_block['end'] = int(row['next_timestep'])
            else:
                # Commit completed block and start a new one
                compressed_records.append(current_block)
                current_block = {
                    'intersection_id': intersection_id,
                    'phase_name': row['phase_name'],
                    'phase': row['phase'],
                    'start': int(row['timestep']),
                    'end': int(row['next_timestep'])
                }
        # Append the final block
        compressed_records.append(current_block)
        
    df_compressed = pd.DataFrame(compressed_records)
    # Calculate continuous duration for rendering lengths
    df_compressed['duration'] = df_compressed['end'] - df_compressed['start']
    
    print(f"Data Compression optimized: Reduced {len(df_raw)} raw log entries down to {len(df_compressed)} visualization states.")
    return df_compressed


def generate_pipeline_timeline(df, selected_intersections=None):
    """
    Generates an interactive Plotly horizontal timeline chart.
    """
    # Filter intersections if a subset is explicitly requested
    if selected_intersections:
        df = df[df['intersection_id'].isin(selected_intersections)]
    
    # Sort intersections to maintain a clean vertical sequence
    df = df.sort_values('intersection_id')
    
    # Create timeline using a horizontal bar chart with custom 'base' settings
    fig = px.bar(
        df,
        x="duration",
        base="start",
        y="intersection_id",
        color="phase_name",
        orientation="h",
        text="phase_name",
        hover_data={
            "intersection_id": True,
            "phase_name": True,
            "phase": True,
            "start": True,
            "end": True,
            "duration": False # redundant with start/end
        },
        labels={
            "intersection_id": "Intersection ID",
            "phase_name": "Signal Phase",
            "duration": "Timesteps"
        }
    )
    
    # Force bars to overlay at their exact base coordinates instead of stacking
    fig.update_layout(barmode="overlay")
    
    # Optimize layout and aesthetic presentations
    fig.update_layout(
        title="Traffic Signal Phase Timeline Evolution",
        xaxis_title="Simulation Timestep",
        yaxis_title="Intersections",
        xaxis=dict(type='linear', sharpends=True),
        yaxis=dict(autorange="reversed"), # Top-down ordering matching log hierarchy
        legend_title_text="Phase Names",
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    # Clean text inside the timeline bars
    fig.update_traces(textposition='inside', insidetextanchor='middle')
    
    return fig


# ==========================================
# TEST IMPLEMENTATION (MOCK DATA GENERATION)
# ==========================================
if __name__ == "__main__":
    # Generate a realistic mock JSON log string representing multiple intersections over 200 timesteps
    mock_log_data = {
        "original_run_details": {"author": "SUMO_Sim", "duration_expected": 200}
    }
    
    # Simulate 3 intersections cycling phases dynamically
    intersections = ["intersection_1_1", "intersection_1_2", "intersection_2_1"]
    phases_pool = [
        {"name": "ETWT_GREEN", "str": "rrrryyrrrrrryyrr"},
        {"name": "ETWT_ALL_RED", "str": "rrrrrrrrrrrrrrrr"},
        {"name": "NSSB_GREEN", "str": "ggyygg ggyygg"},
    ]
    
    current_phase_idx = {int_id: 0 for int_id in intersections}
    
    # Build logs at alternating time steps to test sparseness and compression
    for ts in range(0, 201, 5):
        mock_log_data[str(ts)] = []
        for int_id in intersections:
            # Change phase every 30 timesteps per intersection, offset slightly
            hash_mod = (ts + (intersections.index(int_id) * 15)) % 45
            if hash_mod == 0:
                current_phase_idx[int_id] = (current_phase_idx[int_id] + 1) % len(phases_pool)
            
            chosen_phase = phases_pool[current_phase_idx[int_id]]
            
            mock_log_data[str(ts)].append({
                "intersection_id": int_id,
                "phase": chosen_phase["str"],
                "phase_name": chosen_phase["name"]
            })
            
    # Standardize to raw JSON string mimicking your file input
    json_payload = json.dumps(mock_log_data)
    
    # Run parsing and visualization pipeline
    processed_df = load_and_compress_traffic_logs(json_payload, is_string=True)
    
    # Optional selection filter test: (e.g., ['intersection_1_1', 'intersection_1_2'])
    fig = generate_pipeline_timeline(processed_df, selected_intersections=None)
    
    # Open visualization natively inside default web browser
    fig.show()