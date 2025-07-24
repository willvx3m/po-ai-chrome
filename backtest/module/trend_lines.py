def calculate_trend_lines(candles):
    """
    Calculate the best support and resistance lines that minimize the area between them.
    Support line must be strictly below all candle lows, and resistance line must be
    strictly above all candle highs.

    Args:
        candles: List of dictionaries with keys 'x' (index), 'low', 'high' representing candlestick data.

    Returns:
        dict: Contains support and resistance line equations and the minimum area.
    """
    # Extract low, and high values
    points = [(index, c["low"], c["high"]) for index, c in enumerate(candles)]
    n = len(points)
    if n < 2:
        return {"error": "Need at least two candles to calculate trend lines"}

    # Identify potential support, resistance points
    support_points = []
    resistance_points = []
    for i in range(n):
        x, low, high = points[i]
        # Skip local-low high points for resistance line
        if (i == 0 or high <= points[i - 1][2]) and (
            i == n - 1 or high <= points[i + 1][2]
        ):
            pass
        else:
            resistance_points.append((x, high))
        # Skip local-high low points for support line
        if (i == 0 or low >= points[i - 1][1]) and (
            i == n - 1 or low >= points[i + 1][1]
        ):
            pass
        else:
            support_points.append((x, low))

    if len(support_points) < 2 or len(resistance_points) < 2:
        return {"error": "Not enough lower lows or higher highs to form lines"}
    
    # Function to calculate line equation (slope, intercept)
    def get_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        if x2 == x1:
            return None  # Avoid division by zero
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept

    # Function to check if support line is strictly below all candle lows
    def is_valid_support(slope, intercept, points):
        for x, low, _ in points:
            if slope * x + intercept >= low:
                return False
        return True

    # Function to check if resistance line is strictly above all candle highs
    def is_valid_resistance(slope, intercept, points):
        for x, _, high in points:
            if slope * x + intercept <= high:
                return False
        return True

    # Function to calculate area between two lines over [x_min, x_max]
    def calculate_area(slope_r, intercept_r, slope_s, intercept_s, x_min, x_max):
        m_diff = slope_r - slope_s
        b_diff = intercept_r - intercept_s
        area = (m_diff / 2) * (x_max**2 - x_min**2) + b_diff * (x_max - x_min)
        return abs(area)

    # Initialize variables
    min_area = float("inf")
    best_support = None
    best_resistance = None
    x_min = 0
    x_max = len(candles) - 1

    # Small offset to ensure lines are strictly below/above
    epsilon = 1e-10  # Adjust as needed for precision

    # Iterate through all possible support and resistance line pairs
    valid_pairs = []
    for i in range(len(support_points)):
        for j in range(i + 1, len(support_points)):
            support_line = get_line(support_points[i], support_points[j])
            if support_line is None:
                continue
            slope_s, intercept_s = support_line
            # Shift support line slightly below the anchor points
            intercept_s -= epsilon
            if not is_valid_support(slope_s, intercept_s, points):
                continue

            for k in range(len(resistance_points)):
                for l in range(k + 1, len(resistance_points)):
                    resistance_line = get_line(resistance_points[k], resistance_points[l])
                    if resistance_line is None:
                        continue
                    slope_r, intercept_r = resistance_line
                    # Shift resistance line slightly above the anchor points
                    intercept_r += epsilon
                    if not is_valid_resistance(slope_r, intercept_r, points):
                        continue
                        
                    # print("Valid resistance line", slope_r, intercept_r, resistance_points[k], resistance_points[l])

                    # Ensure resistance is above support for all x
                    valid_pair = True
                    for x, _, _ in points:
                        if slope_r * x + intercept_r <= slope_s * x + intercept_s:
                            valid_pair = False
                            break
                    if not valid_pair:
                        continue

                    area = calculate_area(
                        slope_r, intercept_r, slope_s, intercept_s, x_min, x_max
                    )
                    valid_pairs.append(
                        (
                            area,
                            slope_s,
                            intercept_s,
                            support_points[i],
                            support_points[j],
                            slope_r,
                            intercept_r,
                            resistance_points[k],
                            resistance_points[l],
                        )
                    )

                    # print("Valid pair", area, slope_s, intercept_s, slope_r, intercept_r, support_points[i], support_points[j], resistance_points[k], resistance_points[l])

    if not valid_pairs:
        return {
            "error": "No valid support and resistance lines found with strict constraints"
        }

    # Find the pair with minimum area
    min_area, slope_s, intercept_s, p1_s, p2_s, slope_r, intercept_r, p1_r, p2_r = min(
        valid_pairs, key=lambda x: x[0]
    )

    # Format result
    return {
        "support": {
            "slope": slope_s,
            "intercept": intercept_s,
            "points": [p1_s, p2_s],
            "equation": f"y = {slope_s:.6f}x + {intercept_s:.6f}",
        },
        "resistance": {
            "slope": slope_r,
            "intercept": intercept_r,
            "points": [p1_r, p2_r],
            "equation": f"y = {slope_r:.6f}x + {intercept_r:.6f}",
        },
        "area": min_area,
    }


def main():
    candles = [
        {
            "time_label": "12:32",
            "x": 15,
            "open": 1.1284884080370943,
            "close": 1.1282411128284389,
            "high": 1.1285069551777434,
            "low": 1.128179289026275,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:32",
        },
        {
            "time_label": "12:33",
            "x": 42,
            "open": 1.1285564142194744,
            "close": 1.1281916537867078,
            "high": 1.1285564142194744,
            "low": 1.1281916537867078,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:33",
        },
        {
            "time_label": "12:34",
            "x": 68,
            "open": 1.128500772797527,
            "close": 1.1283523956723338,
            "high": 1.128500772797527,
            "low": 1.1280309119010818,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:34",
        },
        {
            "time_label": "12:35",
            "x": 94,
            "open": 1.1288469860896444,
            "close": 1.1283647604327665,
            "high": 1.1288469860896444,
            "low": 1.1283647604327665,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:35",
        },
        {
            "time_label": "12:36",
            "x": 121,
            "open": 1.128840803709428,
            "close": 1.1283462132921174,
            "high": 1.1288840803709428,
            "low": 1.1283462132921174,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:36",
        },
        {
            "time_label": "12:37",
            "x": 147,
            "open": 1.128420401854714,
            "close": 1.1281236476043275,
            "high": 1.128420401854714,
            "low": 1.1281236476043275,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:37",
        },
        {
            "time_label": "12:38",
            "x": 173,
            "open": 1.128340030911901,
            "close": 1.1278639876352394,
            "high": 1.1283833075734158,
            "low": 1.1278639876352394,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:38",
        },
        {
            "time_label": "12:39",
            "x": 200,
            "open": 1.1281112828438948,
            "close": 1.1276228748068005,
            "high": 1.1281112828438948,
            "low": 1.1276228748068005,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:39",
        },
        {
            "time_label": "12:40",
            "x": 226,
            "open": 1.128389489953632,
            "close": 1.1275734157650694,
            "high": 1.1284265842349304,
            "low": 1.1275734157650694,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:40",
        },
        {
            "time_label": "12:41",
            "x": 252,
            "open": 1.1285069551777434,
            "close": 1.127919629057187,
            "high": 1.1285069551777434,
            "low": 1.127919629057187,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:41",
        },
        {
            "time_label": "12:42",
            "x": 278,
            "open": 1.1281112828438948,
            "close": 1.1278825347758887,
            "high": 1.1281112828438948,
            "low": 1.1278825347758887,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:42",
        },
        {
            "time_label": "12:43",
            "x": 305,
            "open": 1.1283771251931993,
            "close": 1.127919629057187,
            "high": 1.1283771251931993,
            "low": 1.127919629057187,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:43",
        },
        {
            "time_label": "12:44",
            "x": 331,
            "open": 1.12829057187017,
            "close": 1.128129829984544,
            "high": 1.1283523956723338,
            "low": 1.1281112828438948,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:44",
        },
        {
            "time_label": "12:45",
            "x": 358,
            "open": 1.1282040185471405,
            "close": 1.1279938176197835,
            "high": 1.1282040185471405,
            "low": 1.1279938176197835,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:45",
        },
        {
            "time_label": "12:46",
            "x": 384,
            "open": 1.1285687789799073,
            "close": 1.128098918083462,
            "high": 1.1285873261205563,
            "low": 1.1280803709428129,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:46",
        },
        {
            "time_label": "12:47",
            "x": 410,
            "open": 1.1290942812982998,
            "close": 1.1284884080370943,
            "high": 1.1290942812982998,
            "low": 1.1284884080370943,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:47",
        },
        {
            "time_label": "12:48",
            "x": 437,
            "open": 1.1293292117465223,
            "close": 1.1289829984544049,
            "high": 1.129372488408037,
            "low": 1.128952086553323,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:48",
        },
        {
            "time_label": "12:49",
            "x": 463,
            "open": 1.1296568778979907,
            "close": 1.1293106646058733,
            "high": 1.1296568778979907,
            "low": 1.1293106646058733,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:49",
        },
        {
            "time_label": "12:50",
            "x": 489,
            "open": 1.129341576506955,
            "close": 1.128970633693972,
            "high": 1.12945285935085,
            "low": 1.128970633693972,
            "date": "2025-05-09",
            "datetime_point": "2025-05-09 12:50",
        },
    ]
    result = calculate_trend_lines(candles)
    print(result)
    if result.get("support"):
        print(result.get("support").get("equation"))
    if result.get("resistance"):
        print(result.get("resistance").get("equation"))
    if result.get("area"):
        print(result.get("area"))

# if __name__ == "__main__":
#     main()