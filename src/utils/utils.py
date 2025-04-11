import os
import pandas as pd
from typing import Literal
import json
import random
from pathlib import Path
from typing import List, Dict

## 用来分割数据集 ##

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



def split_train_val(
    dataset: str,
    train: str,
    val: str,
    split_ratio: float = 0.8
) -> None:
    """
    Splits a JSON file containing an array of objects into two files with the specified ratio.

    Parameters:
    - dataset (str): Path to the input JSON file.
    - train (str): Path to save the train dataset 
    - val (str): Path to save the validation dataset
    - split_ratio (float): Proportion of data to include in the first split (train). Default is 0.8 (80%).

    Returns:
    - None
    """
    try:
        with open(dataset, "r") as file:
            data: List[Dict] = json.load(file)  

        random.shuffle(data)

        split_point: int = int(len(data) * split_ratio)
        train_data: List[Dict] = data[:split_point]
        val_data: List[Dict] = data[split_point:]

        # Save the split data to separate files
        with open(train, "w") as file:
            json.dump(train_data, file, indent=4)

        with open(val, "w") as file:
            json.dump(val_data, file, indent=4)

        print(f"Data split completed!")
        print(f"{len(train_data)} records saved to {train}")
        print(f"{len(val_data)} records saved to {val}")
    
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    #merge_csv_files("../eval/data/ZhiCheng_MCQ_A1.csv", "../eval/data/ZhiCheng_MCQ_A2.csv", "../eval/data/mcq.csv", merge_type="row")
    
    split_train_val("../../data/DSv2sum_2270_dataset.json","../../data/WeDoctor/train.json","../../data/WeDoctor/val.json")



if __name__ == "__main__":
    main()
    