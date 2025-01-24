import pandas as pd
from typing import Literal

def merge_csv_files(
    file1: str,
    file2: str,
    output_file: str,
    merge_type: Literal["row", "column"] = "row"
) -> None:
    """
    Merges two CSV files either row-wise or column-wise and saves the result to a new file.

    Parameters:
    - file1 (str): Path to the first CSV file.
    - file2 (str): Path to the second CSV file.
    - output_file (str): Path to save the merged CSV file.
    - merge_type (Literal["row", "column"]): Type of merge ('row' for row-wise, 'column' for column-wise).
      Defaults to 'row'.

    Returns:
    - None
    """
    try:
        # Read the CSV files
        df1: pd.DataFrame = pd.read_csv(file1)
        df2: pd.DataFrame = pd.read_csv(file2)
        
        merged_df: pd.DataFrame
        if merge_type == "row":
            # Merge row-wise
            merged_df = pd.concat([df1, df2], ignore_index=True)
        elif merge_type == "column":
            # Merge column-wise
            merged_df = pd.concat([df1, df2], axis=1)
        else:
            raise ValueError("Invalid merge_type. Use 'row' or 'column'.")
        
        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv(output_file, index=False)
        print(f"Merged CSV file saved to: {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")



def main():
    merge_csv_files("../eval/data/ZhiCheng_MCQ_A1.csv", "../eval/data/ZhiCheng_MCQ_A2.csv", "../eval/data/mcq.csv", merge_type="row")



if __name__ == "__main__":
    main()
    