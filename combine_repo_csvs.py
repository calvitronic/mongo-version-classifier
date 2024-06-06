import os
import pandas as pd

def combine_csv_files(directory_path, output_file):
    # List to hold individual dataframes
    csv_files = []
    
    # Iterate through all files in the directory
    for file_name in os.listdir(directory_path):
        # Create the full file path
        file_path = os.path.join(directory_path, file_name)
        if file_name.endswith('.csv') and (os.path.getsize(file_path) > 10):
            # Read the CSV file and append to the list
            csv_files.append(pd.read_csv(file_path))
    
    # Concatenate all dataframes
    combined_csv = pd.concat(csv_files, ignore_index=True)
    
    # Save the combined dataframe to a new CSV file
    combined_csv.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

# Example usage
directory_path = '../all_versions/csvs'
output_file = 'combined_repos_output.csv'
combine_csv_files(directory_path, output_file)
