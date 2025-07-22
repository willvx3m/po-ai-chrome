#!/usr/bin/env python3
"""
Script to complete missing candle data in merged.json by averaging nearby candles.
Finds gaps in the datetime sequence and fills them with interpolated candles.
"""

import json
import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import argparse


def parse_datetime(datetime_str: str) -> datetime.datetime:
    """Parse datetime string in format 'YYYY-MM-DD HH:MM'"""
    return datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")


def format_datetime(dt: datetime.datetime) -> str:
    """Format datetime to 'YYYY-MM-DD HH:MM'"""
    return dt.strftime("%Y-%m-%d %H:%M")


def format_time_label(dt: datetime.datetime) -> str:
    """Format datetime to 'HH:MM' time label"""
    return dt.strftime("%H:%M")


def find_missing_datetime_points(candles: List[Dict[str, Any]]) -> List[datetime.datetime]:
    """
    Find missing datetime points in the candle data.
    
    Args:
        candles: List of candle dictionaries with 'datetime_point' field
        
    Returns:
        List of missing datetime points
    """
    if not candles:
        return []
    
    # Sort candles by datetime_point
    sorted_candles = sorted(candles, key=lambda x: parse_datetime(x['datetime_point']))
    
    missing_points = []
    
    # Check for gaps between consecutive candles
    for i in range(len(sorted_candles) - 1):
        current_dt = parse_datetime(sorted_candles[i]['datetime_point'])
        next_dt = parse_datetime(sorted_candles[i + 1]['datetime_point'])
        
        # Calculate expected next datetime (1 minute later)
        expected_next = current_dt + datetime.timedelta(minutes=1)
        
        # If there's a gap, add missing points
        if next_dt > expected_next:
            current = expected_next
            while current < next_dt:
                missing_points.append(current)
                current += datetime.timedelta(minutes=1)
    
    return missing_points


def interpolate_candle(prev_candle: Dict[str, Any], next_candle: Dict[str, Any], 
                      target_dt: datetime.datetime) -> Dict[str, Any]:
    """
    Create an interpolated candle by averaging the previous and next candles.
    
    Args:
        prev_candle: Previous candle data
        next_candle: Next candle data
        target_dt: Target datetime for the interpolated candle
        
    Returns:
        Interpolated candle dictionary
    """
    # Average the OHLC values
    open_val = (prev_candle['open'] + next_candle['open']) / 2
    close_val = (prev_candle['close'] + next_candle['close']) / 2
    high_val = (prev_candle['high'] + next_candle['high']) / 2
    low_val = (prev_candle['low'] + next_candle['low']) / 2
    
    # Interpolate x coordinate (linear interpolation)
    prev_dt = parse_datetime(prev_candle['datetime_point'])
    next_dt = parse_datetime(next_candle['datetime_point'])
    
    # Calculate time ratio for interpolation
    total_diff = (next_dt - prev_dt).total_seconds()
    target_diff = (target_dt - prev_dt).total_seconds()
    ratio = target_diff / total_diff if total_diff > 0 else 0.5
    
    x_val = prev_candle['x'] + (next_candle['x'] - prev_candle['x']) * ratio
    
    return {
        'time_label': format_time_label(target_dt),
        'x': x_val,
        'open': open_val,
        'close': close_val,
        'high': high_val,
        'low': low_val,
        'date': target_dt.strftime('%Y-%m-%d'),
        'datetime_point': format_datetime(target_dt)
    }


def find_nearest_candles(candles: List[Dict[str, Any]], target_dt: datetime.datetime, 
                        window_minutes: int = 5) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    """
    Find the nearest candles before and after the target datetime.
    
    Args:
        candles: List of sorted candle dictionaries
        target_dt: Target datetime
        window_minutes: Time window to search for nearby candles
        
    Returns:
        Tuple of (prev_candle, next_candle)
    """
    # Sort candles by datetime
    sorted_candles = sorted(candles, key=lambda x: parse_datetime(x['datetime_point']))
    
    # Find candles within the window
    window_start = target_dt - datetime.timedelta(minutes=window_minutes)
    window_end = target_dt + datetime.timedelta(minutes=window_minutes)
    
    nearby_candles = []
    for candle in sorted_candles:
        candle_dt = parse_datetime(candle['datetime_point'])
        if window_start <= candle_dt <= window_end:
            nearby_candles.append(candle)
    
    if len(nearby_candles) < 2:
        # If not enough candles in window, use the closest ones
        min_distance = float('inf')
        closest_before = None
        closest_after = None
        
        for candle in sorted_candles:
            candle_dt = parse_datetime(candle['datetime_point'])
            distance = abs((candle_dt - target_dt).total_seconds())
            
            if candle_dt < target_dt and (closest_before is None or distance < min_distance):
                closest_before = candle
                min_distance = distance
            elif candle_dt > target_dt and (closest_after is None or distance < min_distance):
                closest_after = candle
                min_distance = distance
        
        return closest_before, closest_after
    
    # Find closest before and after within window
    before_candles = [c for c in nearby_candles if parse_datetime(c['datetime_point']) < target_dt]
    after_candles = [c for c in nearby_candles if parse_datetime(c['datetime_point']) > target_dt]
    
    prev_candle = max(before_candles, key=lambda x: parse_datetime(x['datetime_point'])) if before_candles else None
    next_candle = min(after_candles, key=lambda x: parse_datetime(x['datetime_point'])) if after_candles else None
    
    return prev_candle, next_candle


def complete_missing_candles(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Complete missing candles by interpolating between existing ones.
    
    Args:
        candles: List of candle dictionaries
        
    Returns:
        List of candles with missing data filled in
    """
    if not candles:
        return candles
    
    # Find missing datetime points
    missing_points = find_missing_datetime_points(candles)
    
    if not missing_points:
        print("✅ No missing datetime points found!")
        return candles
    
    print(f"Found {len(missing_points)} missing datetime points")
    
    # Create new candles for missing points
    new_candles = []
    
    for missing_dt in missing_points:
        prev_candle, next_candle = find_nearest_candles(candles, missing_dt)
        
        if prev_candle and next_candle:
            interpolated_candle = interpolate_candle(prev_candle, next_candle, missing_dt)
            new_candles.append(interpolated_candle)
            print(f"  Interpolated candle for {format_datetime(missing_dt)}")
        else:
            print(f"  ⚠️  Could not interpolate candle for {format_datetime(missing_dt)} - insufficient nearby data")
    
    # Combine original and new candles
    all_candles = candles + new_candles
    
    # Sort by datetime
    all_candles.sort(key=lambda x: parse_datetime(x['datetime_point']))
    
    print(f"✅ Added {len(new_candles)} interpolated candles")
    return all_candles


def main():
    """Main function to process the merged.json file."""
    parser = argparse.ArgumentParser(description='Complete missing candle data in merged.json')
    parser.add_argument('--input', default='merged.json', help='Input JSON file (default: merged.json)')
    parser.add_argument('--output', default='merged-complete.json', help='Output JSON file (default: merged-complete.json)')
    parser.add_argument('--window', type=int, default=5, help='Time window in minutes for finding nearby candles (default: 5)')
    
    args = parser.parse_args()
    
    try:
        # Load the JSON file
        print(f"Loading {args.input} file...")
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} candles")
        
        # Complete missing candles
        print("Completing missing candle data...")
        completed_data = complete_missing_candles(data)
        
        # Save the completed data
        print(f"Saving completed data to {args.output}...")
        with open(args.output, 'w') as f:
            json.dump(completed_data, f, indent=2)
        
        print(f"✅ Successfully saved {len(completed_data)} candles to {args.output}")
        
        # Summary
        original_count = len(data)
        final_count = len(completed_data)
        added_count = final_count - original_count
        
        print(f"\n📊 Summary:")
        print(f"  • Original candles: {original_count}")
        print(f"  • Final candles: {final_count}")
        print(f"  • Added candles: {added_count}")
        
    except FileNotFoundError:
        print(f"❌ Error: {args.input} file not found!")
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {args.input}!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main() 