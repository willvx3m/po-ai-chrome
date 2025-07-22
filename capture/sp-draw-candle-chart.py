#!/usr/bin/env python3
"""
Script to draw a scrollable candle chart from JSON data with zoom in/out functionality.
Supports aggregated averages for different timeframes.
"""

import json
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.widgets import TextBox
from typing import List, Dict, Any, Tuple
import tkinter as tk
from tkinter import filedialog


class CandleChart:
    def __init__(self, data: List[Dict[str, Any]]):
        self.original_data = data
        self.current_data = data
        self.aggregation_level = 1  # 1 minute intervals
        self.view_start = 0
        self.max_candles = 60  # Maximum candles to display at once
        self.view_end = min(self.max_candles, len(data))  # Show first 60 candles by default
        self.zoom_factor = 1.0
        self.candle_patches = []  # Store candle patches for hover detection
        self.tooltip = None
        
        # Parse datetime and sort data
        self.parse_and_sort_data()
        
        # Create the plot
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.setup_plot()
        self.setup_controls()
        self.setup_hover_events()
        self.update_chart()
        
    def parse_and_sort_data(self):
        """Parse datetime and sort the data"""
        for candle in self.original_data:
            candle['datetime'] = datetime.datetime.strptime(
                candle['datetime_point'], '%Y-%m-%d %H:%M'
            )
        
        self.original_data.sort(key=lambda x: x['datetime'])
        self.current_data = self.original_data.copy()
    
    def aggregate_data(self, level: int) -> List[Dict[str, Any]]:
        """
        Aggregate data by averaging OHLC values over specified time intervals.
        
        Args:
            level: Aggregation level in minutes (1, 5, 15, 30, 60, etc.)
            
        Returns:
            List of aggregated candle data
        """
        if level == 1:
            return self.original_data
        
        aggregated = []
        current_group = []
        
        for candle in self.original_data:
            if not current_group:
                current_group.append(candle)
                continue
            
            # Check if this candle belongs to the same time group
            time_diff = (candle['datetime'] - current_group[0]['datetime']).total_seconds() / 60
            
            if time_diff < level:
                current_group.append(candle)
            else:
                # Aggregate the current group
                if current_group:
                    agg_candle = self.aggregate_candle_group(current_group)
                    aggregated.append(agg_candle)
                
                # Start new group
                current_group = [candle]
        
        # Don't forget the last group
        if current_group:
            agg_candle = self.aggregate_candle_group(current_group)
            aggregated.append(agg_candle)
        
        return aggregated
    
    def aggregate_candle_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate a group of candles into a single candle"""
        if not group:
            return {}
        
        # Calculate OHLC for the group
        opens = [c['open'] for c in group]
        highs = [c['high'] for c in group]
        lows = [c['low'] for c in group]
        closes = [c['close'] for c in group]
        
        # OHLC calculation
        open_val = opens[0]  # First open
        close_val = closes[-1]  # Last close
        high_val = max(highs)  # Highest high
        low_val = min(lows)  # Lowest low
        
        # Use the first candle's datetime and other metadata
        base_candle = group[0]
        
        return {
            'time_label': base_candle['time_label'],
            'x': base_candle['x'],
            'open': open_val,
            'close': close_val,
            'high': high_val,
            'low': low_val,
            'date': base_candle['date'],
            'datetime_point': base_candle['datetime_point'],
            'datetime': base_candle['datetime']
        }
    
    def setup_plot(self):
        """Setup the main plot"""
        self.ax.set_title('Candle Chart (Scrollable with Zoom)', fontsize=14)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Price')
        self.ax.grid(True, alpha=0.3)
        
        # Format x-axis - use simple integer ticks instead of datetime
        self.ax.set_xlabel('Time')
        self.ax.tick_params(axis='x', rotation=45)
    
    def setup_controls(self):
        """Setup interactive controls"""
        # Create control panel
        plt.subplots_adjust(bottom=0.25, left=0.1, right=0.9)
        
        # View range slider
        max_start = max(0, len(self.original_data) - self.max_candles)
        ax_slider = plt.axes((0.1, 0.15, 0.65, 0.03))
        self.slider = Slider(
            ax_slider, 'View Position', 0, 
            max_start, 
            valinit=0, valstep=10
        )
        self.slider.on_changed(self.on_slider_change)
        
        # Zoom slider
        ax_zoom = plt.axes((0.1, 0.1, 0.65, 0.03))
        self.zoom_slider = Slider(
            ax_zoom, 'Zoom', 0.5, 2.0, 
            valinit=1.0, valstep=0.1
        )
        self.zoom_slider.on_changed(self.on_zoom_change)
        
        # Aggregation buttons
        ax_agg = plt.axes((0.8, 0.15, 0.15, 0.1))
        self.agg_buttons = RadioButtons(
            ax_agg, ('1m', '5m', '15m', '30m', '1h'),
            active=0
        )
        self.agg_buttons.on_clicked(self.on_aggregation_change)
        
        # Navigation buttons
        ax_prev = plt.axes((0.8, 0.05, 0.06, 0.03))
        ax_next = plt.axes((0.89, 0.05, 0.06, 0.03))
        
        self.btn_prev = Button(ax_prev, '←')
        self.btn_next = Button(ax_next, '→')
        
        self.btn_prev.on_clicked(self.on_prev_click)
        self.btn_next.on_clicked(self.on_next_click)
    
    def setup_hover_events(self):
        """Setup hover events for tooltips"""
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.fig.canvas.mpl_connect('axes_leave_event', self.on_leave)
    
    def on_slider_change(self, val):
        """Handle slider change"""
        self.view_start = int(val)
        self.view_end = min(self.view_start + int(self.max_candles / self.zoom_factor), len(self.current_data))
        self.update_chart()
    
    def on_zoom_change(self, val):
        """Handle zoom change"""
        self.zoom_factor = val
        view_width = int(self.max_candles / self.zoom_factor)
        self.view_end = min(self.view_start + view_width, len(self.current_data))
        self.update_chart()
    
    def on_aggregation_change(self, label):
        """Handle aggregation level change"""
        agg_levels = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60}
        self.aggregation_level = agg_levels[label]
        self.current_data = self.aggregate_data(self.aggregation_level)
        
        # Reset view
        self.view_start = 0
        self.view_end = min(self.max_candles, len(self.current_data))
        self.update_chart()
    
    def on_prev_click(self, event):
        """Handle previous button click"""
        step = int(10 / self.zoom_factor)
        self.view_start = max(0, self.view_start - step)
        self.view_end = min(self.view_start + int(self.max_candles / self.zoom_factor), len(self.current_data))
        self.slider.set_val(self.view_start)
        self.update_chart()
    
    def on_next_click(self, event):
        """Handle next button click"""
        step = int(10 / self.zoom_factor)
        self.view_start = min(len(self.current_data) - int(self.max_candles / self.zoom_factor), 
                             self.view_start + step)
        self.view_end = min(self.view_start + int(self.max_candles / self.zoom_factor), len(self.current_data))
        self.slider.set_val(self.view_start)
        self.update_chart()
    
    def on_hover(self, event):
        """Handle mouse hover events"""
        if event.inaxes != self.ax:
            return
        
        # Find which candle is being hovered
        x_pos = int(event.xdata)
        if 0 <= x_pos < len(self.candle_patches):
            candle = self.current_data[self.view_start + x_pos]
            self.show_tooltip(event, candle)
    
    def on_leave(self, event):
        """Handle mouse leave events"""
        self.hide_tooltip()
    
    def show_tooltip(self, event, candle: Dict[str, Any]):
        """Show tooltip with candle data"""
        if self.tooltip:
            self.tooltip.remove()
        
        # Create tooltip text
        tooltip_text = f"Time: {candle['time_label']}\n"
        tooltip_text += f"Date: {candle['date']}\n"
        tooltip_text += f"Open: {candle['open']:.6f}\n"
        tooltip_text += f"High: {candle['high']:.6f}\n"
        tooltip_text += f"Low: {candle['low']:.6f}\n"
        tooltip_text += f"Close: {candle['close']:.6f}"
        
        # Position tooltip near mouse
        x, y = event.x, event.y
        self.tooltip = self.ax.text(x, y, tooltip_text, 
                                   bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
                                   fontsize=8, transform=self.ax.transData)
        self.fig.canvas.draw_idle()
    
    def hide_tooltip(self):
        """Hide tooltip"""
        if self.tooltip:
            self.tooltip.remove()
            self.tooltip = None
            self.fig.canvas.draw_idle()
    
    def draw_candlestick(self, candle: Dict[str, Any], x_pos: int):
        """Draw a single candlestick"""
        open_val = candle['open']
        close_val = candle['close']
        high_val = candle['high']
        low_val = candle['low']
        
        # Determine color based on open/close
        color = 'green' if close_val >= open_val else 'red'
        edge_color = 'darkgreen' if close_val >= open_val else 'darkred'
        
        # Draw the wick (high-low line)
        self.ax.plot([x_pos, x_pos], [low_val, high_val], 
                    color=edge_color, linewidth=1)
        
        # Draw the body
        body_height = abs(close_val - open_val)
        body_bottom = min(open_val, close_val)
        
        rect = Rectangle((x_pos - 0.3, body_bottom), 0.6, body_height,
                        facecolor=color, edgecolor=edge_color, linewidth=1)
        self.ax.add_patch(rect)
        
        # Store the patch for hover detection
        self.candle_patches.append(rect)
    
    def update_chart(self):
        """Update the chart display"""
        self.ax.clear()
        self.setup_plot()
        
        # Clear previous candle patches
        self.candle_patches = []
        
        # Get the current view data
        view_data = self.current_data[self.view_start:self.view_end]
        
        if not view_data:
            return
        
        # Draw candlesticks
        for i, candle in enumerate(view_data):
            self.draw_candlestick(candle, i)
        
        # Set x-axis limits
        self.ax.set_xlim(-1, len(view_data))
        
        # Set x-axis ticks to show time labels
        if len(view_data) > 20:
            tick_step = max(1, len(view_data) // 10)
            tick_positions = range(0, len(view_data), tick_step)
            tick_labels = [view_data[i]['time_label'] for i in tick_positions]
        else:
            tick_positions = range(len(view_data))
            tick_labels = [candle['time_label'] for candle in view_data]
        
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(tick_labels)
        
        # Set y-axis limits with some padding
        all_prices = []
        for candle in view_data:
            all_prices.extend([candle['open'], candle['close'], candle['high'], candle['low']])
        
        if all_prices:
            min_price = min(all_prices)
            max_price = max(all_prices)
            price_range = max_price - min_price
            padding = price_range * 0.05
            
            self.ax.set_ylim(min_price - padding, max_price + padding)
        
        # Update title with current info
        if view_data:
            start_time = view_data[0]['datetime']
            end_time = view_data[-1]['datetime']
            agg_text = f" ({self.aggregation_level}m)" if self.aggregation_level > 1 else ""
            
            self.ax.set_title(
                f'Candle Chart{agg_text} - {start_time.strftime("%Y-%m-%d %H:%M")} to {end_time.strftime("%H:%M")} '
                f'(Showing {len(view_data)} candles, Position {self.view_start}-{self.view_end})',
                fontsize=12
            )
        
        # Update slider range
        max_start = max(0, len(self.current_data) - int(self.max_candles / self.zoom_factor))
        self.slider.valmax = max_start
        self.slider.ax.set_xlim(0, max_start)
        
        plt.draw()


def load_json_data(filename: str) -> List[Dict[str, Any]]:
    """Load JSON data from file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} candles from {filename}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: {filename} file not found!")
        return []
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {filename}!")
        return []
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return []


def select_file() -> str:
    """Open file dialog to select JSON file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    filename = filedialog.askopenfilename(
        title="Select JSON file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    
    return filename if filename else "merged-complete.json"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Draw scrollable candle chart from JSON data')
    parser.add_argument('--input', help='Input JSON file (default: merged-complete.json)')
    parser.add_argument('--file-dialog', action='store_true', 
                       help='Open file dialog to select input file')
    
    args = parser.parse_args()
    
    # Determine input file
    if args.file_dialog:
        input_file = select_file()
        if not input_file:
            print("No file selected. Exiting.")
            return
    elif args.input:
        input_file = args.input
    else:
        input_file = "merged-complete.json"
    
    # Load data
    data = load_json_data(input_file)
    if not data:
        return
    
    # Create and display chart
    print("Creating interactive candle chart...")
    print("Controls:")
    print("  - Slider: Navigate through time")
    print("  - Zoom slider: Zoom in/out")
    print("  - Radio buttons: Change aggregation level")
    print("  - Arrow buttons: Quick navigation")
    print("  - Mouse wheel: Zoom in/out")
    print("  - Mouse drag: Pan around")
    
    chart = CandleChart(data)
    plt.show()


if __name__ == "__main__":
    main() 