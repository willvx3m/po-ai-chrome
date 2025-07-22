#!/usr/bin/env python3
"""
Script to check for missing datetime points in merged.json file.
Finds gaps in the datetime sequence and displays them as ranges.
"""

import json
import datetime
from typing import List, Tuple, Dict, Any
from collections import defaultdict


def parse_datetime(datetime_str: str) -> datetime.datetime:
    """Parse datetime string in format 'YYYY-MM-DD HH:MM'"""
    return datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")


def format_datetime(dt: datetime.datetime) -> str:
    """Format datetime to 'YYYY-MM-DD HH:MM'"""
    return dt.strftime("%Y-%m-%d %H:%M")


def find_missing_datetime_ranges(candles: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Find missing datetime points in the candle data.
    
    Args:
        candles: List of candle dictionaries with 'datetime_point' field
        
    Returns:
        List of tuples containing (start_datetime, end_datetime) ranges
    """
    if not candles:
        return []
    
    # Sort candles by datetime_point
    sorted_candles = sorted(candles, key=lambda x: parse_datetime(x['datetime_point']))
    
    missing_ranges = []
    
    # Check for gaps between consecutive candles
    for i in range(len(sorted_candles) - 1):
        current_dt = parse_datetime(sorted_candles[i]['datetime_point'])
        next_dt = parse_datetime(sorted_candles[i + 1]['datetime_point'])
        
        # Calculate expected next datetime (1 minute later)
        expected_next = current_dt + datetime.timedelta(minutes=1)
        
        # If there's a gap, add it to missing ranges
        if next_dt > expected_next:
            missing_start = format_datetime(expected_next)
            missing_end = format_datetime(next_dt - datetime.timedelta(minutes=1))
            missing_ranges.append((missing_start, missing_end))
    
    return missing_ranges


def group_missing_ranges_by_date(missing_ranges: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Group missing ranges by date for better organization.
    
    Args:
        missing_ranges: List of (start, end) datetime tuples
        
    Returns:
        Dictionary with date as key and list of time ranges as value
    """
    grouped = defaultdict(list)
    
    for start_dt, end_dt in missing_ranges:
        # Extract date from datetime string
        date = end_dt.split()[0]
        # Extract time ranges
        start_time = start_dt.split()[1]
        end_time = end_dt.split()[1]
        
        grouped[date].append((start_time, end_time))
    
    return dict(grouped)


def main():
    """Main function to process the merged.json file."""
    try:
        # Load the JSON file
        print("Loading merged.json file...")
        with open('merged.json', 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} candles")
        
        # Find missing datetime ranges
        print("Analyzing datetime sequence...")
        missing_ranges = find_missing_datetime_ranges(data)
        
        if not missing_ranges:
            print("✅ No missing datetime points found!")
            return
        
        print(f"\n❌ Found {len(missing_ranges)} missing datetime ranges:")
        print("=" * 60)
        
        # Group by date for better display
        grouped_ranges = group_missing_ranges_by_date(missing_ranges)
        
        for date in sorted(grouped_ranges.keys()):
            print(f"\n📅 Date: {date}")
            print("-" * 40)
            
            for start_time, end_time in grouped_ranges[date]:
                if start_time == end_time:
                    print(f"  ⏰ Missing: {start_time}")
                    pass
                else:
                    print(f"  ⏰ Missing: {start_time} -> {end_time}")
        
        # Summary statistics
        total_missing_minutes = sum(
            (parse_datetime(end) - parse_datetime(start)).total_seconds() / 60 + 1
            for start, end in missing_ranges
        )
        
        print(f"\n📊 Summary:")
        print(f"  • Total missing ranges: {len(missing_ranges)}")
        print(f"  • Total missing minutes: {int(total_missing_minutes)}")
        print(f"  • Affected dates: {len(grouped_ranges)}")
        
    except FileNotFoundError:
        print("❌ Error: merged.json file not found!")
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON format in merged.json!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
