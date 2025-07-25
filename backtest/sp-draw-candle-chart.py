#!/usr/bin/env python3
"""
Script to draw a scrollable candle chart from complete.json with zoom in/out functionality.
Supports aggregated averages for different timeframes and trading position visualization.

Usage:
python sp-draw-candle-chart.py --input complete.json
python sp-draw-candle-chart.py --file-dialog
python sp-draw-candle-chart.py --input complete.json --trading-results results.json

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
from module.trend_lines import calculate_trend_lines
from module.indicator import calculate_ema_indicator, calculate_sma_indicator

# Indicator period constants
EMA_LONG_PERIOD = 30
EMA_SHORT_PERIOD = 5
SMA_PERIOD = 5

class CandleChart:
    def __init__(self, data: List[Dict[str, Any]], trading_results: List[Dict[str, Any]] = None):
        self.original_data = data
        self.current_data = data
        self.trading_results = trading_results or []
        self.aggregation_level = 1  # 1 minute intervals
        self.view_start = 0
        self.max_candles = 60  # Maximum candles to display at once
        self.view_end = min(self.max_candles, len(data))  # Show first 60 candles by default
        self.zoom_factor = 1.0
        self.tooltip = None
        self.trend_lines = []  # Store drawn trend lines
        self.show_trend_lines = True  # Toggle for showing trend lines
        self.show_trading_positions = True  # Toggle for showing trading positions
        
        # Indicator toggles
        self.show_ema_long = False
        self.show_ema_short = False
        self.show_sma = False
        
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
        
        # Format x-axis
        self.ax.set_xlabel('Time')
        self.ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    def setup_controls(self):
        """Setup interactive controls"""
        # Create control panel with more space for x-axis labels
        plt.subplots_adjust(bottom=0.2, left=0.1, right=0.9, top=0.9)
        
        # View range slider
        max_start = max(0, len(self.original_data) - self.max_candles)
        ax_slider = plt.axes((0.1, 0.1, 0.65, 0.03))
        self.slider = Slider(
            ax_slider, 'View Position', 0, 
            max_start, 
            valinit=0, valstep=10
        )
        self.slider.on_changed(self.on_slider_change)
        
        # Zoom slider
        ax_zoom = plt.axes((0.1, 0.05, 0.65, 0.03))
        self.zoom_slider = Slider(
            ax_zoom, 'Zoom', 0.5, 2.0, 
            valinit=1.0, valstep=0.1
        )
        self.zoom_slider.on_changed(self.on_zoom_change)
        
        # Aggregation buttons
        ax_agg = plt.axes((0.8, 0.04, 0.1, 0.1))
        self.agg_buttons = RadioButtons(
            ax_agg, ('1m', '5m', '15m', '30m', '1h'),
            active=0
        )
        self.agg_buttons.on_clicked(self.on_aggregation_change)
        
        # Navigation buttons
        ax_prev = plt.axes((0.82, 0.9, 0.04, 0.03))
        ax_next = plt.axes((0.86, 0.9, 0.04, 0.03))
        
        self.btn_prev = Button(ax_prev, '←')
        self.btn_next = Button(ax_next, '→')
        
        self.btn_prev.on_clicked(self.on_prev_click)
        self.btn_next.on_clicked(self.on_next_click)
        
        # Trend line control buttons
        ax_draw_trend = plt.axes((0.92, 0.9, 0.06, 0.03))
        ax_erase_trend = plt.axes((0.92, 0.85, 0.06, 0.03))
        
        self.btn_draw_trend = Button(ax_draw_trend, 'Draw Trend')
        self.btn_erase_trend = Button(ax_erase_trend, 'Erase Trend')
        
        self.btn_draw_trend.on_clicked(self.on_draw_trend_click)
        self.btn_erase_trend.on_clicked(self.on_erase_trend_click)
        
        # Trading positions toggle button
        if self.trading_results:
            ax_toggle_positions = plt.axes((0.91, 0.8, 0.08, 0.03))
            self.btn_toggle_positions = Button(ax_toggle_positions, 'Toggle Positions')
            self.btn_toggle_positions.on_clicked(self.on_toggle_positions_click)
            
            # Trading position navigation buttons
            ax_prev_trade = plt.axes((0.91, 0.75, 0.04, 0.03))
            ax_next_trade = plt.axes((0.95, 0.75, 0.04, 0.03))
            
            self.btn_prev_trade = Button(ax_prev_trade, '← Trade')
            self.btn_next_trade = Button(ax_next_trade, 'Trade →')
            
            self.btn_prev_trade.on_clicked(self.on_prev_trade_click)
            self.btn_next_trade.on_clicked(self.on_next_trade_click)
        
        # Indicator toggle buttons
        ax_ema_50 = plt.axes((0.91, 0.7, 0.08, 0.03))
        ax_ema_10 = plt.axes((0.91, 0.66, 0.08, 0.03))
        ax_sma_10 = plt.axes((0.91, 0.62, 0.08, 0.03))
        
        self.btn_ema_50 = Button(ax_ema_50, f'EMA({EMA_LONG_PERIOD})')
        self.btn_ema_10 = Button(ax_ema_10, f'EMA({EMA_SHORT_PERIOD})')
        self.btn_sma_10 = Button(ax_sma_10, f'SMA({SMA_PERIOD})')
        
        self.btn_ema_50.on_clicked(self.on_toggle_ema_50_click)
        self.btn_ema_10.on_clicked(self.on_toggle_ema_10_click)
        self.btn_sma_10.on_clicked(self.on_toggle_sma_10_click)
    
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
        # Clear trend lines when zooming
        self.trend_lines = []
        self.update_chart()
    
    def on_aggregation_change(self, label):
        """Handle aggregation level change"""
        agg_levels = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60}
        self.aggregation_level = agg_levels[label]
        self.current_data = self.aggregate_data(self.aggregation_level)
        
        # Reset view and clear trend lines
        self.view_start = 0
        self.view_end = min(self.max_candles, len(self.current_data))
        self.trend_lines = []
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
    
    def on_draw_trend_click(self, event):
        """Handle draw trend button click"""
        view_data = self.current_data[self.view_start:self.view_end]
        if len(view_data) >= 2:
            result = calculate_trend_lines(view_data)
            support = result.get("support")
            resistance = result.get("resistance")
            self.trend_lines = []
            if support:
                slope_s, intercept_s = support.get("slope"), support.get("intercept")
                self.trend_lines.append((
                    (self.view_start, intercept_s),
                    (self.view_end, slope_s * len(view_data) + intercept_s),
                    'support'
                ))
            if resistance:
                slope_r, intercept_r = resistance.get("slope"), resistance.get("intercept")
                self.trend_lines.append((
                    (self.view_start, intercept_r),
                    (self.view_end, slope_r * len(view_data) + intercept_r),
                    'resistance'
                ))
            self.update_chart()
    
    def on_erase_trend_click(self, event):
        """Handle erase trend button click"""
        self.trend_lines = []
        self.update_chart()
    
    def on_toggle_positions_click(self, event):
        """Handle toggle positions button click"""
        self.show_trading_positions = not self.show_trading_positions
        self.update_chart()
    
    def on_toggle_ema_50_click(self, event):
        """Handle toggle EMA(50) button click"""
        self.show_ema_long = not self.show_ema_long
        self.update_chart()
    
    def on_toggle_ema_10_click(self, event):
        """Handle toggle EMA(10) button click"""
        self.show_ema_short = not self.show_ema_short
        self.update_chart()
    
    def on_toggle_sma_10_click(self, event):
        """Handle toggle SMA(10) button click"""
        self.show_sma = not self.show_sma
        self.update_chart()
    
    def get_all_trading_positions(self):
        """Get all trading positions sorted by entry time"""
        positions = []
        for result in self.trading_results:
            strategy_start = result.get('strategy_start_index', 0)
            strategy_finished = result.get('strategy_finished_index', 0)
            positions_list = result.get('positions', [])
            profit = result.get('profit', 0)
            
            for position in positions_list:
                entry_candle_index = position.get('entry_candle_index', 0)
                positions.append({
                    'entry_index': entry_candle_index,
                    'exit_index': strategy_finished,
                    'position': position,
                    'profit': profit,
                    'strategy_start': strategy_start,
                    'strategy_finished': strategy_finished
                })
        
        # Sort by entry index
        positions.sort(key=lambda x: x['entry_index'])
        return positions
    
    def find_nearest_trading_position(self, current_index: int, direction: str = 'next'):
        """Find the nearest trading position in the specified direction"""
        positions = self.get_all_trading_positions()
        if not positions:
            return None
        
        if direction == 'next':
            # Find the next position after current_index
            for pos in positions:
                if pos['entry_index'] > current_index:
                    return pos
            return None
        else:  # previous
            # Find the previous position before current_index
            for pos in reversed(positions):
                if pos['entry_index'] < current_index:
                    return pos
            return None
    
    def center_view_on_position(self, position_data: Dict[str, Any]):
        """Center the view on a specific trading position"""
        entry_index = position_data['entry_index']
        exit_index = position_data['exit_index']
        
        # Calculate the center point of the position
        center_index = (entry_index + exit_index) // 2
        
        # Calculate view width based on zoom
        view_width = int(self.max_candles / self.zoom_factor)
        
        # Calculate the start index to center the position
        view_start = max(0, center_index - view_width // 2)
        
        # Ensure we don't go beyond the data bounds
        max_start = max(0, len(self.current_data) - view_width)
        view_start = min(view_start, max_start)
        
        # Update the view
        self.view_start = view_start
        self.view_end = min(view_start + view_width, len(self.current_data))
        self.slider.set_val(view_start)
        self.update_chart()
    
    def on_prev_trade_click(self, event):
        """Handle previous trading position button click"""
        if not self.trading_results:
            return
        
        # Find the nearest previous trading position
        current_center = self.view_start + (self.view_end - self.view_start) // 2
        prev_position = self.find_nearest_trading_position(current_center, 'previous')
        
        if prev_position:
            self.center_view_on_position(prev_position)
        else:
            print("No previous trading position found.")
    
    def on_next_trade_click(self, event):
        """Handle next trading position button click"""
        if not self.trading_results:
            return
        
        # Find the nearest next trading position
        current_center = self.view_start + (self.view_end - self.view_start) // 2
        next_position = self.find_nearest_trading_position(current_center, 'next')
        
        if next_position:
            self.center_view_on_position(next_position)
        else:
            print("No next trading position found.")
    
    def on_hover(self, event):
        """Handle mouse hover events"""
        if event.inaxes != self.ax:
            self.hide_tooltip()
            return
        
        # Get current view data
        view_data = self.current_data[self.view_start:self.view_end]
        if not view_data:
            return
        
        # Find which candle is being hovered based on x position
        x_pos = event.xdata
        if x_pos is None or x_pos < 0 or x_pos >= len(view_data):
            self.hide_tooltip()
            return
        
        candle_index = int(x_pos)
        if 0 <= candle_index < len(view_data):
            candle = view_data[candle_index]
            self.show_tooltip(event, candle)
        else:
            self.hide_tooltip()
    
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
        
        # Position tooltip near mouse but ensure it's visible
        x_data, y_data = event.xdata, event.ydata
        if x_data is not None and y_data is not None:
            # Convert data coordinates to display coordinates
            x_display, y_display = self.ax.transData.transform((x_data, y_data))
            
            # Add offset to position tooltip above the candle
            y_offset = 20
            self.tooltip = self.ax.text(x_data, y_data, tooltip_text, 
                                       bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.95),
                                       fontsize=8, transform=self.ax.transData,
                                       verticalalignment='bottom')
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
    
    def draw_trading_position(self, position: Dict[str, Any], x_pos: int, y_pos: float, duration_end_x: int = None):
        """
        Draw a single trading position marker with duration.
        
        Args:
            position: Dictionary containing position data (direction, amount, entry_price, etc.)
            x_pos: X position on the chart (entry point)
            y_pos: Y position (price level)
            duration_end_x: X position where the position ends (exit point)
        """
        direction = position.get('direction', 'call')
        amount = position.get('amount', 1)
        entry_price = position.get('entry_price', y_pos)
        profit = position.get('profit', 0)  # If available in the position data
        
        # Determine marker properties based on direction and profit
        if direction == 'call':
            marker = '^'  # Upward triangle for call
            color = 'green' if profit >= 0 else 'red'
        else:  # put
            marker = 'v'  # Downward triangle for put
            color = 'green' if profit >= 0 else 'red'
        
        # Adjust marker size based on amount
        marker_size = 8 + (amount * 2)  # Base size + amount multiplier
        
        # Draw the position marker
        self.ax.scatter(x_pos, entry_price, 
                       marker=marker, 
                       s=marker_size, 
                       c=color, 
                       edgecolors='black', 
                       linewidth=1,
                       alpha=0.8,
                       zorder=5)  # Ensure markers are above candles
        
        # Add text label for amount
        self.ax.text(x_pos + 0.5, entry_price, 
                    f'{amount}', 
                    fontsize=8, 
                    ha='left', 
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8),
                    zorder=6)
        
        # Draw duration line/rectangle if we have an end point
        if duration_end_x is not None and duration_end_x > x_pos:
            # Draw a horizontal line or rectangle to show position duration
            duration_width = duration_end_x - x_pos
            
            # Create a rectangle to show the position duration
            rect = Rectangle((x_pos, entry_price - 0.0001), duration_width, 0.0002,
                           facecolor=color, alpha=0.3, edgecolor=color, linewidth=1, zorder=4)
            self.ax.add_patch(rect)
            
            # Add exit marker at the end
            exit_marker = 'o'  # Circle for exit point
            self.ax.scatter(duration_end_x, entry_price, 
                           marker=exit_marker, 
                           s=marker_size * 0.7, 
                           c=color, 
                           edgecolors='black', 
                           linewidth=1,
                           alpha=0.8,
                           zorder=5)
    
    def draw_strategy_profit(self, strategy_finished_index: int, profit: float, y_pos: float):
        """
        Draw profit text after the last position of a strategy.
        
        Args:
            strategy_finished_index: The index where the strategy finished
            profit: The profit/loss of the strategy
            y_pos: Y position (price level) for the text
        """
        # Calculate relative position within the view
        view_start_global = self.view_start
        relative_x = strategy_finished_index - view_start_global
        
        # Only draw if the strategy end is within the current view
        if 0 <= relative_x < len(self.current_data[self.view_start:self.view_end]):
            # Determine color based on profit
            color = 'green' if profit >= 0 else 'red'
            
            # Format profit text
            profit_text = f"{profit:+.2f}"
            
            # Draw profit text
            self.ax.text(relative_x + 1, y_pos, 
                        profit_text, 
                        fontsize=10, 
                        ha='left', 
                        va='center',
                        color=color,
                        weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor=color),
                        zorder=7)  # Ensure text is above everything else
    
    def draw_trend_lines_from_json(self, trend_lines: Dict[str, Any], trend_start_index: int, trend_end_index: int):
        """
        Draw trend lines from JSON data.
        
        Args:
            trend_lines: Dictionary containing support and resistance line data
            trend_start_index: Start index of the trend lines
            trend_end_index: End index of the trend lines
        """
        if not trend_lines or "error" in trend_lines:
            return
        
        # Calculate relative positions within the view
        view_start_global = self.view_start
        view_end_global = self.view_start + len(self.current_data[self.view_start:self.view_end])
        
        # Check if trend lines overlap with current view
        if trend_start_index >= view_end_global or trend_end_index <= view_start_global:
            return
        
        # Calculate relative start and end positions
        relative_start = trend_start_index - view_start_global
        relative_end = trend_end_index - view_start_global

        # Draw support line
        if "support" in trend_lines:
            support = trend_lines["support"]
            slope_s = support.get("slope", 0)
            intercept_s = support.get("intercept", 0)
            
            # Calculate y values for start and end points
            y_start_support = intercept_s
            y_end_support = slope_s * (relative_end - relative_start) + intercept_s
            
            # Draw support line
            self.ax.plot([relative_start, relative_end], [y_start_support, y_end_support], 
                        color='blue', linestyle='--', linewidth=2, alpha=0.7, zorder=3, label='Support')
        
        # Draw resistance line
        if "resistance" in trend_lines:
            resistance = trend_lines["resistance"]
            slope_r = resistance.get("slope", 0)
            intercept_r = resistance.get("intercept", 0)
            
            # Calculate y values for start and end points
            y_start_resistance = intercept_r
            y_end_resistance = slope_r * (relative_end - relative_start) + intercept_r
            
            # Draw resistance line
            self.ax.plot([relative_start, relative_end], [y_start_resistance, y_end_resistance], 
                        color='orange', linestyle='--', linewidth=2, alpha=0.7, zorder=3, label='Resistance')
    
    def calculate_indicator_values(self, indicator_type: str, period: int) -> List[float]:
        """
        Calculate indicator values for the current view.
        
        Args:
            indicator_type: 'ema' or 'sma'
            period: period for the indicator
            
        Returns:
            List of indicator values for the current view
        """
        view_data = self.current_data[self.view_start:self.view_end]
        values = []
        
        for i in range(len(view_data)):
            # Get candles up to current position
            candles_up_to_current = self.current_data[max(0, self.view_start+i+1-period):self.view_start + i + 1]
            
            if indicator_type == 'ema':
                value = calculate_ema_indicator(candles_up_to_current, period)
            elif indicator_type == 'sma':
                value = calculate_sma_indicator(candles_up_to_current, period)
            else:
                value = None
            
            values.append(value)
        
        return values
    
    def draw_indicators(self):
        """Draw all enabled indicators"""
        view_data = self.current_data[self.view_start:self.view_end]
        if not view_data:
            return
        
        x_positions = list(range(len(view_data)))
        
        # Draw EMA(50)
        if self.show_ema_long:
            ema_long_values = self.calculate_indicator_values('ema', EMA_LONG_PERIOD)
            valid_ema_long = [(i, val) for i, val in enumerate(ema_long_values) if val is not None]
            if valid_ema_long:
                x_vals, y_vals = zip(*valid_ema_long)
                self.ax.plot(x_vals, y_vals, color='purple', linewidth=2, alpha=0.8, 
                           label=f'EMA({EMA_LONG_PERIOD})', zorder=2)
        
        # Draw EMA(10)
        if self.show_ema_short:
            ema_short_values = self.calculate_indicator_values('ema', EMA_SHORT_PERIOD)
            valid_ema_short = [(i, val) for i, val in enumerate(ema_short_values) if val is not None]
            if valid_ema_short:
                x_vals, y_vals = zip(*valid_ema_short)
                self.ax.plot(x_vals, y_vals, color='orange', linewidth=2, alpha=0.8, 
                           label=f'EMA({EMA_SHORT_PERIOD})', zorder=2)
        
        # Draw SMA(10)
        if self.show_sma:
            sma_values = self.calculate_indicator_values('sma', SMA_PERIOD)
            valid_sma = [(i, val) for i, val in enumerate(sma_values) if val is not None]
            if valid_sma:
                x_vals, y_vals = zip(*valid_sma)
                self.ax.plot(x_vals, y_vals, color='blue', linewidth=2, alpha=0.8, 
                           label=f'SMA({SMA_PERIOD})', zorder=2)
        
        # Add legend if any indicators are shown
        if any([self.show_ema_long, self.show_ema_short, self.show_sma]):
            self.ax.legend(loc='upper left', fontsize=8)
    
    def draw_trading_positions(self):
        """Draw all trading positions that fall within the current view"""
        if not self.show_trading_positions or not self.trading_results:
            return
        
        # Get the current view data
        view_data = self.current_data[self.view_start:self.view_end]
        if not view_data:
            return
        
        # Find the global indices for the current view
        view_start_global = self.view_start
        view_end_global = self.view_start + len(view_data)
        
        # Track strategies that have been processed to avoid duplicate profit labels
        processed_strategies = set()
        
        for result in self.trading_results:
            strategy_start = result.get('strategy_start_index', 0)
            strategy_finished = result.get('strategy_finished_index', 0)
            positions = result.get('positions', [])
            profit = result.get('profit', 0)
            
            # Get trend line data from the result
            trend_lines = result.get('trend_lines', None)
            trend_start_index = result.get('trend_start_index', strategy_start)
            trend_end_index = result.get('trend_end_index', strategy_finished)
            
            # Check if this strategy result overlaps with current view
            if (strategy_start <= view_end_global and strategy_finished >= view_start_global):
                
                # Draw trend lines if available
                if trend_lines:
                    self.draw_trend_lines_from_json(trend_lines, trend_start_index, trend_end_index)
                
                # Track if we've drawn any positions for this strategy
                strategy_has_positions_in_view = False
                last_position_y = None
                
                # Draw each position in this strategy
                for position in positions:
                    entry_candle_index = position.get('entry_candle_index', 0)
                    
                    # Check if this position is within the current view or extends into it
                    if (entry_candle_index < view_end_global and strategy_finished >= view_start_global):
                        
                        # Calculate relative positions within the view
                        entry_relative_x = max(0, entry_candle_index - view_start_global)
                        exit_relative_x = min(len(view_data), strategy_finished - view_start_global)
                        
                        # Only draw if the position has some visibility in the current view
                        if entry_relative_x < len(view_data):
                            # Get the entry price
                            entry_price = position.get('entry_price', 0)
                            last_position_y = entry_price  # Track for profit label placement
                            
                            # Add profit information to position data for coloring
                            position_with_profit = position.copy()
                            position_with_profit['profit'] = profit
                            
                            # Determine the end x position for duration visualization
                            duration_end_x = None
                            if exit_relative_x > entry_relative_x and exit_relative_x <= len(view_data):
                                duration_end_x = exit_relative_x
                            
                            # Draw the position with duration
                            self.draw_trading_position(position_with_profit, entry_relative_x, entry_price, duration_end_x)
                            strategy_has_positions_in_view = True
                
                # Draw profit label if this strategy has positions in view and hasn't been processed
                if (strategy_has_positions_in_view and 
                    strategy_finished not in processed_strategies and 
                    last_position_y is not None):
                    
                    # Use the last position's price level for profit label placement
                    self.draw_strategy_profit(strategy_finished, profit, last_position_y)
                    processed_strategies.add(strategy_finished)
    
    def update_chart(self):
        """Update the chart display"""
        self.ax.clear()
        self.setup_plot()
        
        # Get the current view data
        view_data = self.current_data[self.view_start:self.view_end]
        
        if not view_data:
            return
        
        # Draw candlesticks
        for i, candle in enumerate(view_data):
            self.draw_candlestick(candle, i)
        
        # Draw trading positions
        self.draw_trading_positions()
        
        # Draw indicators
        self.draw_indicators()
        
        # Set x-axis limits
        self.ax.set_xlim(-1, len(view_data))
        
        # Set x-axis ticks to show time labels
        if len(view_data) > 20:
            tick_step = max(1, len(view_data) // 20)
            tick_positions = range(0, len(view_data), tick_step)
            tick_labels = [view_data[i]['time_label'] for i in tick_positions]
        else:
            tick_positions = range(len(view_data))
            tick_labels = [candle['time_label'] for candle in view_data]
        
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(tick_labels)
        
        # Improve x-axis label visibility
        self.ax.tick_params(axis='x', rotation=45, labelsize=8)
        self.ax.tick_params(axis='y', labelsize=9)
        
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
            positions_text = " + Positions" if self.show_trading_positions and self.trading_results else ""
            
            # Add indicator text to title
            indicators = []
            if self.show_ema_long:
                indicators.append(f"EMA({EMA_LONG_PERIOD})")
            if self.show_ema_short:
                indicators.append(f"EMA({EMA_SHORT_PERIOD})")
            if self.show_sma:
                indicators.append(f"SMA({SMA_PERIOD})")
            
            indicators_text = " + " + ", ".join(indicators) if indicators else ""
            
            self.ax.set_title(
                f'Candle Chart{agg_text}{positions_text}{indicators_text} - {start_time.strftime("%Y-%m-%d %H:%M")} to {end_time.strftime("%Y-%m-%d %H:%M")} '
                f'(Showing {len(view_data)} candles, Position {self.view_start}-{self.view_end})',
                fontsize=12
            )
        
        # Update slider range
        max_start = max(0, len(self.current_data) - int(self.max_candles / self.zoom_factor))
        self.slider.valmax = max_start
        self.slider.ax.set_xlim(0, max_start)
        
        # Draw trend lines if they exist
        if self.trend_lines:
            for index, trend_line in enumerate(self.trend_lines):
                # trend_line is a list of points
                x_coords = [trend_line[0][0] - self.view_start, trend_line[1][0] - self.view_start]
                y_coords = [trend_line[0][1], trend_line[1][1]]
                line_type = trend_line[2]
                color = 'blue' if line_type == 'support' else 'orange'
                label = 'Support' if line_type == 'support' else 'Resistance'
                self.ax.plot(x_coords, y_coords, color=color, linestyle='--', linewidth=2, label=label)
            
            # Add legend for trend lines
            self.ax.legend(loc='upper left', fontsize=8)
        
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


def load_trading_results(filename: str) -> List[Dict[str, Any]]:
    """Load trading results JSON data from file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} trading results from {filename}")
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
    
    return filename if filename else "complete.json"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Draw scrollable candle chart from JSON data')
    parser.add_argument('--input', help='Input JSON file (default: complete.json)')
    parser.add_argument('--file-dialog', action='store_true', 
                       help='Open file dialog to select input file')
    parser.add_argument('--trading-results', help='Trading results JSON file to overlay positions')
    
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
        input_file = "complete.json"
    
    # Load candle data
    data = load_json_data(input_file)
    if not data:
        return
    
    # Load trading results if provided
    trading_results = []
    if args.trading_results:
        trading_results = load_trading_results(args.trading_results)
        if not trading_results:
            print("Warning: Could not load trading results, continuing without position overlay.")
    
    # Create and display chart
    print("Creating interactive candle chart...")
    print("Controls:")
    print("  - Slider: Navigate through time")
    print("  - Zoom slider: Zoom in/out")
    print("  - Radio buttons: Change aggregation level")
    print("  - Arrow buttons: Quick navigation")
    print("  - Mouse wheel: Zoom in/out")
    print("  - Mouse drag: Pan around")
    print("  - EMA LONG, EMA SHORT, SMA: Toggle technical indicators")
    if trading_results:
        print("  - Toggle Positions: Show/hide trading positions")
        print("  - ← Trade / Trade →: Navigate between trading positions")
    
    chart = CandleChart(data, trading_results)
    plt.show()


if __name__ == "__main__":
    main() 