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

    print(f"Shape of unprocessed csv: {combined_csv.shape}")
    
        # Select only numeric columns for variance calculation
    numeric_columns = combined_csv.select_dtypes(include=['number']).columns
    print(f"Number of 'numeric columns': {len(numeric_columns)}")
    # numeric_columns_index_set = set(numeric_columns.index)

    # csv_index_set = set(combined_csv.columns)

    # numeric_columns_index_set = set(numeric_columns)

    #print(f"Combined csv index set: {csv_index_set}")

    numeric_data = combined_csv[numeric_columns]
    
    # Calculate variance of each numeric column
    variances = numeric_data.var()
    
    # Filter out numeric columns with variance below the threshold
    numeric_columns_to_keep_master = variances[variances > variance_threshold]
    numeric_columns_to_keep = numeric_columns_to_keep_master.index
    # print(f"Columns to not keep: {numeric_columns_to_keep}")

    # numeric_columns_to_keep_set = set(numeric_columns_to_keep_master)

    # print(f"Numeric_columns_to_keep_set: {numeric_columns_to_keep_set}")

    print(f"Number of 'numeric columns to keep': {len(numeric_columns_to_keep)}")
    
    # Combine the numeric columns to keep with non-numeric columns
    non_numeric_columns = combined_csv.select_dtypes(exclude=['number']).columns

    print(f"Number of 'non-numeric columns': {len(non_numeric_columns)}")
    # print(f"Non-numeric columns labels: {non_numeric_columns}")

    columns_to_keep = numeric_columns_to_keep.union(non_numeric_columns)
    
    filtered_csv = combined_csv[columns_to_keep]

    print(f"Shape of resulting csv: {filtered_csv.shape}")

    # filtered_csv_index_set = set(filtered_csv.columns)

    # print(f"Excluded numeric columns: {list(filtered_csv_index_set - numeric_columns_to_keep_set)}")

    # print(f"Excluded indic(es): {list(csv_index_set - filtered_csv_index_set)}")
    
    # Save the filtered dataframe to a new CSV file
    filtered_csv.to_csv(output_file, index=False)
    print(f"Filtered combined CSV saved to {output_file}")
    # combined_csv.to_csv(output_file, index=False)
    # print(f"Combined CSV saved to {output_file}")

# Example usage
variance_threshold = 0.1
# directory_path = '../testtestversionmongoallbaseversionsnetcond3/csvs'
# output_file = 'combined_repos_output_varied_network_conditions_no_removals.csv'

# directory_path = '../all_versions/csvs'
# output_file = 'combined_repos_output_varied_no_network_conditions_no_removals.csv'

directory_path = '../General_CSVS'
output_file = 'combined_net_cond_and_no_net_cond_remove_5_var.csv'

combine_csv_files(directory_path, output_file)
