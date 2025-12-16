import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===== CONFIGURATION - Change these values =====
# EDGE_NUMBER can be either:
#   - A single number: EDGE_NUMBER = 11
#   - A list: EDGE_NUMBER = [11, 12, 13]
#   - A range: EDGE_NUMBER = range(11, 15)
EDGE_NUMBER = 54

# EDGE_DATA_DAY can be either:
#   - A single number: EDGE_DATA_DAY = 2  (1 = day3, 2 = day4, etc.)
#   - A list: EDGE_DATA_DAY = [1, 2, 3]
#   - A range: EDGE_DATA_DAY = range(1, 6)
EDGE_DATA_DAY = range(1, 6)

# USE_PROCESSED_DATA: Switch between raw and processed data
#   - True: Use processed data from output/ folder
#   - False: Use raw data from data/ folder
USE_PROCESSED_DATA = True

# COMPARE_MODE: Plot both raw and processed data for comparison
#   - True: Show both raw (data/) and processed (output/) on the same graph
#   - False: Show only the dataset selected by USE_PROCESSED_DATA
COMPARE_MODE = False

# TIME_RANGE: Filter data to specific time range
#   - Set to None to show full day: TIME_START = None, TIME_END = None
#   - Set to specific times: TIME_START = "10:35", TIME_END = "12:05"
TIME_START = None  # Format: "HH:MM" or None for start of day
TIME_END = None    # Format: "HH:MM" or None for end of day

# CONTINUOUS_DAY_PLOT: Plot data as continuous days on x-axis
#   - True: Plot multiple days continuously with actual dates on x-axis
#   - False: Use default mode (each day overlaps on same time scale)
CONTINUOUS_DAY_PLOT = True

# START_DATE: Starting date for continuous day plot (only used if CONTINUOUS_DAY_PLOT = True)
#   - Format: "MM/DD/YYYY" or "YYYY-MM-DD"
#   - This will be the date for the first day in EDGE_DATA_DAY
START_DATE = "1/3/2015"
# PLOT_RAW_AVERAGE: Plot rolling average on top of raw data
#   - True: Show a smoothed rolling average line on top of raw data
#   - False: Show only the raw data as-is
PLOT_RAW_AVERAGE = False

# RAW_AVERAGE_WINDOW: Window size for rolling average (number of data points)
#   - Larger values = smoother line, smaller values = follows raw data more closely
RAW_AVERAGE_WINDOW = 20

# LIMIT_Y_AXIS: Limit the y-axis to a specific range
#   - True: Limit y-axis to range specified by Y_AXIS_MIN and Y_AXIS_MAX
#   - False: Auto-scale y-axis based on data
LIMIT_Y_AXIS = True
Y_AXIS_MIN = -0.25
Y_AXIS_MAX = 0.25
# ================================================

# Determine if we're plotting single or multiple edges
if isinstance(EDGE_NUMBER, (list, range)):
    edge_numbers = list(EDGE_NUMBER)
else:
    edge_numbers = [EDGE_NUMBER]

# Determine if we're plotting single or multiple days
if isinstance(EDGE_DATA_DAY, (list, range)):
    data_days = list(EDGE_DATA_DAY)
else:
    data_days = [EDGE_DATA_DAY]

# Parse start date for continuous day plot
if CONTINUOUS_DAY_PLOT:
    try:
        start_date = pd.to_datetime(START_DATE)
    except:
        print(f"Error: Could not parse START_DATE '{START_DATE}'. Using default format.")
        start_date = pd.to_datetime("2025-03-01")

# Create the plot
plt.figure(figsize=(12, 6))

# Track earliest and latest times for x-axis limits
earliest_time = None
latest_time = None

# Determine which folders to use
if COMPARE_MODE:
    data_folders = [("data", "Raw"), ("output", "Processed")]
else:
    folder = "output" if USE_PROCESSED_DATA else "data"
    data_folders = [(folder, None)]

# Create a color map for each edge/day combination
# Generate colors for each unique edge/day combination
color_map = {}
color_cycle = plt.cm.tab10.colors  # Use matplotlib's default color cycle
color_idx = 0
for day_num in data_days:
    for edge_num in edge_numbers:
        key = (day_num, edge_num)
        color_map[key] = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1

# Plot each dataset
for data_folder, dataset_label in data_folders:
    # Collect data from all days first (for continuous rolling average)
    all_days_data = {}  # Dictionary to store combined data per edge

    # First pass: read and combine all days' data
    for day_num in data_days:
        # Construct the CSV file path
        csv_file = f"{data_folder}/edge_data_day{day_num + 2}.csv"

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Convert time_slot to datetime
            df['time_slot'] = pd.to_datetime(df['time_slot'])

            # For continuous day plot, replace dates with actual dates based on START_DATE
            if CONTINUOUS_DAY_PLOT:
                # Calculate the actual date for this day_num
                # First day in data_days gets START_DATE, subsequent days increment
                day_offset = day_num - data_days[0]
                actual_date = start_date + pd.Timedelta(days=day_offset)

                # Replace the date part while keeping the time part
                df['time_slot'] = df['time_slot'].apply(
                    lambda x: actual_date.replace(hour=x.hour, minute=x.minute,
                                                   second=x.second, microsecond=x.microsecond)
                )

            # Filter by time range if specified
            if TIME_START is not None or TIME_END is not None:
                # Get the date from the first row
                date_str = df['time_slot'].iloc[0].strftime('%Y-%m-%d')

                if TIME_START is not None:
                    start_datetime = pd.to_datetime(f"{date_str} {TIME_START}")
                    df = df[df['time_slot'] >= start_datetime]

                if TIME_END is not None:
                    end_datetime = pd.to_datetime(f"{date_str} {TIME_END}")
                    df = df[df['time_slot'] <= end_datetime]

            # Skip if no data after filtering
            if len(df) == 0:
                print(f"Warning: No data in time range for day {day_num + 2}. Skipping.")
                continue

            # Update time limits
            if earliest_time is None or df['time_slot'].iloc[0] < earliest_time:
                earliest_time = df['time_slot'].iloc[0]
            if latest_time is None or df['time_slot'].iloc[-1] > latest_time:
                latest_time = df['time_slot'].iloc[-1]

            # Store data for each edge to combine across days
            for edge_num in edge_numbers:
                edge_column = f'edge{edge_num}_traversal_time_sec'

                # Check if the edge exists in the data
                if edge_column not in df.columns:
                    continue

                # Initialize list for this edge if not exists
                if edge_num not in all_days_data:
                    all_days_data[edge_num] = []

                # Store day_num and dataframe slice for this edge
                all_days_data[edge_num].append({
                    'day_num': day_num,
                    'time_slot': df['time_slot'].values,
                    'data': df[edge_column].values
                })

        except FileNotFoundError:
            print(f"Warning: File {csv_file} not found. Skipping day {day_num + 2}.")
            continue

    # Calculate rolling averages on combined data if needed
    rolling_averages = {}
    # Calculate for raw data (either in compare mode or when using raw data in non-compare mode)
    if PLOT_RAW_AVERAGE and (dataset_label == "Raw" or (not COMPARE_MODE and not USE_PROCESSED_DATA)):
        for edge_num in edge_numbers:
            if edge_num in all_days_data:
                # Combine all days' data for this edge into a single dataframe
                all_data_list = []
                for day_data in all_days_data[edge_num]:
                    day_df = pd.DataFrame({
                        'time_slot': day_data['time_slot'],
                        'value': day_data['data']
                    })
                    all_data_list.append(day_df)

                # Concatenate all days and sort by time
                combined_df = pd.concat(all_data_list, ignore_index=True)
                combined_df = combined_df.sort_values('time_slot').reset_index(drop=True)

                # Calculate rolling average on the combined continuous data
                combined_df['rolling_avg'] = combined_df['value'].rolling(
                    window=RAW_AVERAGE_WINDOW, center=True
                ).mean()

                # Store as a series indexed by time
                rolling_averages[edge_num] = pd.Series(
                    combined_df['rolling_avg'].values,
                    index=pd.to_datetime(combined_df['time_slot'])
                )

    # Second pass: plot everything
    for edge_num in edge_numbers:
        if edge_num not in all_days_data:
            print(f"Warning: Edge {edge_num} not found in any day data. Skipping.")
            continue

        # Plot raw data for each day
        for day_data in all_days_data[edge_num]:
            day_num = day_data['day_num']

            # Create label based on mode
            if COMPARE_MODE:
                if len(data_days) == 1 and len(edge_numbers) == 1:
                    label = f'{dataset_label}'
                elif len(data_days) == 1:
                    label = f'Edge {edge_num} ({dataset_label})'
                else:
                    label = f'Day {day_num + 2}, Edge {edge_num} ({dataset_label})'
            else:
                if len(data_days) == 1:
                    label = f'Edge {edge_num}'
                else:
                    label = f'Day {day_num + 2}, Edge {edge_num}'

            # Different line styles for comparison mode
            if COMPARE_MODE:
                color = color_map[(day_num, edge_num)]
                if dataset_label == "Raw":
                    plt.plot(day_data['time_slot'], day_data['data'], linewidth=1.5,
                            linestyle='--', alpha=0.7, label=label, color=color)
                else:  # Processed
                    plt.plot(day_data['time_slot'], day_data['data'], linewidth=2,
                            linestyle='-', label=label, color=color)
            else:
                # Use day-specific color with muted appearance for raw data
                color = color_map[(day_num, edge_num)]
                plt.plot(day_data['time_slot'], day_data['data'], linewidth=1,
                        linestyle='-', alpha=0.3, label=label, color=color)

        # Plot the continuous rolling average segmented by day with day-specific colors
        if edge_num in rolling_averages:
            rolling_avg = rolling_averages[edge_num]

            # Plot each day's segment of the rolling average with its own color
            for idx, day_data in enumerate(all_days_data[edge_num]):
                day_num = day_data['day_num']
                color = color_map[(day_num, edge_num)]

                # Get the time range for this day (convert to pandas Timestamp for proper comparison)
                day_start = pd.Timestamp(day_data['time_slot'][0])
                day_end = pd.Timestamp(day_data['time_slot'][-1])

                # For connecting segments, extend slightly into next day (except for last day)
                if idx < len(all_days_data[edge_num]) - 1:
                    # Not the last day - include data up to and including first point of next day
                    next_day_start = pd.Timestamp(all_days_data[edge_num][idx + 1]['time_slot'][0])
                    day_avg = rolling_avg[(rolling_avg.index >= day_start) & (rolling_avg.index <= next_day_start)]
                else:
                    # Last day - just use the day's range
                    day_avg = rolling_avg[(rolling_avg.index >= day_start) & (rolling_avg.index <= day_end)]

                # Create label (only for the first segment to avoid duplicate legend entries)
                if day_num == data_days[0]:
                    avg_label = f'Edge {edge_num} (Avg)' if len(data_days) > 1 or len(edge_numbers) > 1 else 'Average'
                else:
                    avg_label = None  # No label for subsequent segments

                if len(day_avg) > 0:
                    plt.plot(day_avg.index, day_avg.values, linewidth=2.5,
                            linestyle='-', alpha=1.0, label=avg_label, color=color)

# Configure the plot
plt.xlabel('Time Slot', fontsize=12)
plt.ylabel('Traversal Time (seconds)', fontsize=12)

# Set title based on single or multiple edges and days
title_suffix = " (Raw vs Processed)" if COMPARE_MODE else ""

if len(edge_numbers) == 1 and len(data_days) == 1:
    plt.title(f'Edge {edge_numbers[0]}, Day {data_days[0] + 2} - Traversal Time Over Time{title_suffix}', fontsize=14)
elif len(edge_numbers) == 1:
    plt.title(f'Edge {edge_numbers[0]}, Days {data_days[0] + 2}-{data_days[-1] + 2} - Traversal Time Over Time{title_suffix}', fontsize=14)
elif len(data_days) == 1:
    plt.title(f'Edges {edge_numbers[0]}-{edge_numbers[-1]}, Day {data_days[0] + 2} - Traversal Time Over Time{title_suffix}', fontsize=14)
else:
    plt.title(f'Multiple Edges and Days - Traversal Time Over Time{title_suffix}', fontsize=14)

# Always show legend if in compare mode or if plotting multiple series
if COMPARE_MODE or len(edge_numbers) > 1 or len(data_days) > 1:
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small', framealpha=0.9, borderaxespad=0)

plt.grid(True, alpha=0.3)

# Set x-axis formatting based on plot mode
ax = plt.gca()

if CONTINUOUS_DAY_PLOT:
    # For continuous day plot, show dates and times
    # Use appropriate locator based on number of days
    num_days = len(data_days)
    if num_days <= 3:
        # Show ticks every 6 hours for up to 3 days
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    elif num_days <= 7:
        # Show ticks every 12 hours for up to a week
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    else:
        # Show daily ticks for longer periods
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
else:
    # For single day overlay mode, show ticks every 30 minutes
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Set x-axis limits
if earliest_time is not None:
    if CONTINUOUS_DAY_PLOT:
        # For continuous plot, use the full range of data
        start_date = earliest_time
        end_date = latest_time
    else:
        # If time range is specified, use the filtered time range
        # Otherwise, start at 00:00
        if TIME_START is not None or TIME_END is not None:
            start_date = earliest_time
        else:
            start_date = earliest_time.normalize()  # Get midnight of the first day
        end_date = latest_time

    ax.set_xlim(start_date, end_date)

# Set y-axis limits if enabled
if LIMIT_Y_AXIS:
    ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
