from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from trend_lines import calculate_trend_lines
import json
from datetime import datetime

app = Flask(__name__)

COMPLETE_JSON_PATH = './eurusd-full.json'
CACHE_JSON_PATH = './eurusd-cache.json'

# Enable CORS for all routes, allowing requests from any origin
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/calculate-trend-lines', methods=['POST'])
def post_calculate_trend_lines():
    try:
        data = request.get_json()
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        total_minutes = (datetime.strptime(end_time, '%Y-%m-%d %H:%M') - datetime.strptime(start_time, '%Y-%m-%d %H:%M')).total_seconds() / 60
        if total_minutes < 10:
            return jsonify({'error': 'Invalid time range, minimum 10 minutes. Requested: ' + str(total_minutes) + ' minutes'}), 400

        # read candle from json file
        with open(COMPLETE_JSON_PATH, 'r') as f:
            candles = json.load(f)
        
        # filter candles by start_time and end_time
        candles = [c for c in candles if c['datetime_point'] >= start_time and c['datetime_point'] <= end_time]
        # check if enough candles are available
        if len(candles) < total_minutes:
            return jsonify({'error': 'Not enough candles available'}), 400

        trend_lines = calculate_trend_lines(candles)
        return jsonify(trend_lines), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add-price-to-time', methods=['POST'])
def post_add_price_to_time():
    try:
        data = request.get_json()
        price = data.get('price')
        time = data.get('time')
        # add price and time to a cache json file
        # when the cache is ready to aggregate for a minute, add the ohlcv to the target json
        return jsonify({'message': 'Added price to time!', 'price': price, 'time': time}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting server on http://localhost:4000...")
    time.sleep(3)  # Delay to allow setup
    app.run(host='0.0.0.0', port=4000)


# Sample curl request to calculate trend lines
# curl -X POST http://localhost:4000/calculate-trend-lines -H "Content-Type: application/json" -d '{"start_time": "2025-05-09 15:00", "end_time": "2025-05-09 20:00"}'