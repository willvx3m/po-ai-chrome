import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Check if a CSV file path is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <path_to_csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

try:
    # Read the CSV file (assumes a single column of y-values, no header)
    data = pd.read_csv(csv_file, header=None)
    y_values = data[0].values.astype(float)  # Ensure y-values are numeric

    # Validate data
    if len(y_values) < 2:
        print("Error: At least 2 data points are required.")
        sys.exit(1)
    if np.any(np.isnan(y_values)) or np.any(np.isinf(y_values)):
        print("Error: Data contains NaN or infinite values.")
        sys.exit(1)

    # Generate x-values (sequential integers starting from 0)
    x_values = np.arange(len(y_values))

    # Scale x-values to [0, 1] for numerical stability
    x_max = max(x_values) if max(x_values) != 0 else 1
    x_scaled = x_values / x_max
    # Scale y-values by dividing by their maximum absolute value
    y_max = np.max(np.abs(y_values))
    y_scaled = y_values / y_max if y_max != 0 else y_values

    # Try fitting polynomials from degree 1 to min(n-1, 10) to find highest stable degree
    max_degree = min(len(y_values) - 1, 10)  # Limit to degree 10 to avoid instability
    best_poly = None
    best_coefficients = None
    best_degree = 0
    best_error = float('inf')

    for degree in range(1, max_degree + 1):
        try:
            # Fit polynomial on scaled data
            coefficients_scaled = np.polyfit(x_scaled, y_scaled, degree)
            # Adjust coefficients for original scale
            scale_factors = y_max / (x_max ** np.arange(degree, -1, -1))
            coefficients = coefficients_scaled * scale_factors
            poly = np.poly1d(coefficients)

            # Compute fit error (mean absolute error)
            fitted_y = poly(x_values)
            error = np.mean(np.abs(fitted_y - y_values))

            # Update best fit if error is acceptable
            if error < best_error:
                best_poly = poly
                best_coefficients = coefficients
                best_degree = degree
                best_error = error

            print(f"Degree {degree} fit successful, mean absolute error: {error}")

        except np.linalg.LinAlgError:
            print(f"Degree {degree} failed: Numerical instability (SVD did not converge).")
            break
        except Exception as e:
            print(f"Degree {degree} failed: {str(e)}")
            break

    if best_poly is None:
        print("Error: No stable polynomial fit found.")
        sys.exit(1)

    # Compute the derivative of the best polynomial
    derivative_poly = np.polyder(best_poly)

    # Print the polynomial and its derivative
    print("\nBest polynomial (highest stable degree):")
    print(f"Degree: {best_degree}")
    print("Coefficients (highest degree first):", best_coefficients)
    print("\nPolynomial function:")
    print(best_poly)
    print("\nDerivative function:")
    print(derivative_poly)
    print("Mean absolute error of fit:", best_error)

    # Plot the data, fitted polynomial, and derivative
    fig, ax1 = plt.subplots()

    # Plot original data and fitted polynomial on primary y-axis
    ax1.scatter(x_values, y_values, color='blue', label='Data points')
    x_smooth = np.linspace(min(x_values), max(x_values), 100)
    ax1.plot(x_smooth, best_poly(x_smooth), color='red', label=f'Fitted polynomial (degree {best_degree})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y (Data and Polynomial)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Create secondary y-axis for the derivative
    ax2 = ax1.twinx()
    ax2.plot(x_smooth, derivative_poly(x_smooth), color='green', linestyle='--', label='Derivative')
    ax2.set_ylabel('Derivative', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    plt.title('Polynomial Fit and Derivative')
    plt.show()

    # Verify the fit
    fitted_y = best_poly(x_values)
    print("\nFitted y-values:", fitted_y)
    print("Original y-values:", y_values)
    print("\nDerivative at x-values:", derivative_poly(x_values))

except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found.")
    sys.exit(1)
except pd.errors.ParserError:
    print("Error: Invalid CSV format. Ensure the file has a single column of numeric values.")
    sys.exit(1)
except ValueError:
    print("Error: Non-numeric values detected in the CSV file.")
    sys.exit(1)
except Exception as e:
    print(f"Error: An unexpected issue occurred: {str(e)}")
    sys.exit(1)