import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import glob
import argparse


def create_line_chart(csv_file, output_file, force=False):
    # Skip if output file exists and force is False
    if not force and os.path.exists(output_file):
        return

    try:
        # Read the CSV file (no header)
        df = pd.read_csv(csv_file, header=None)

        # Create x-axis as index (0, 1, 2, ...)
        x = range(len(df))
        y = df[0]  # Single column

        # Create a new figure
        plt.figure(figsize=(10, 6))

        # Plot the line
        plt.plot(x, y, label="Values", marker="o")

        # Customize the chart
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.title(f"Line Chart: {os.path.basename(csv_file)}")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"Line chart saved as {output_file}")

    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")


def process_folder(folder_path, force=False):
    try:
        # Find all CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        if not csv_files:
            print(f"No CSV files found in {folder_path}")
            return

        # Process each CSV file
        for csv_file in csv_files:
            # Generate output filename (same name, .png extension)
            output_file = os.path.splitext(csv_file)[0] + ".png"
            create_line_chart(csv_file, output_file, force)

    except Exception as e:
        print(f"Error processing folder {folder_path}: {str(e)}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate line charts from CSV files")
    parser.add_argument("input", help="Input CSV file or folder path")
    parser.add_argument(
        "output", nargs="?", help="Output PNG file (required for single-file mode)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of existing output files",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.output and os.path.isdir(args.input):
        print("Error: Output file should not be specified in folder mode")
        print("Usage:")
        print(
            "  Single file mode: python script.py <input_csv_file> <output_png_file> [--force]"
        )
        print("  Folder mode: python script.py <folder_path> [--force]")
        sys.exit(1)

    if not args.output and not os.path.isdir(args.input):
        print("Error: Output file required for single-file mode")
        print("Usage:")
        print(
            "  Single file mode: python script.py <input_csv_file> <output_png_file> [--force]"
        )
        print("  Folder mode: python script.py <folder_path> [--force]")
        sys.exit(1)

    if args.output:
        # Single file mode
        csv_file_path = args.input
        output_file_path = args.output
        create_line_chart(csv_file_path, output_file_path, args.force)
    else:
        # Folder mode
        folder_path = args.input
        if not os.path.isdir(folder_path):
            print(f"Error: {folder_path} is not a valid directory")
            sys.exit(1)
        process_folder(folder_path, args.force)
