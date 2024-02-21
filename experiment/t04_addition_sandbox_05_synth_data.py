'''.

It was hard to get the Synthetic dataset right, so I wrote that portion in
haskell, output csvs, and am playing here to read those csvs back in.

'''

from lark import Lark, Transformer
import pandas as pd
from datasets import Dataset
from io import StringIO
from lark import Lark, Transformer, v_args
from lark import Lark
import re


##########
# Read Haskell-generated CSV.
#
#   Each column contains a sequence which must be parsed into a list.

pattern = re.compile(r'\((Push [^\)]+)\)|(\bPop\b)|(\bNullOp\b)|([A-Z]+)|(\d+)')

def parse_cell(cell):
    # Using findall method to find all matches, ensuring sequences of letters are captured
    matches = pattern.findall(cell)
    # Flattening the results and removing empty strings
    parsed_elements = [''.join(match).strip() for match in matches if ''.join(match).strip()]
    return parsed_elements

def read_csv(data_path):
    df = pd.read_csv(data_path, sep="|")
    df = df[['Input', 'Output', 'PreGlobalOp', 'PreWorkOp', 'PostGlobalOp', 'PostWorkOp']]
    for col in df.columns:
        df[col] = df[col].apply(parse_cell)
    return df

data_path = "experiment/t04_addition/mod_sum_length_3.csv"
data_path = "experiment/t04_addition/mod_sum_length_5.csv"
data_path = "experiment/t04_addition/mod_sum_length_10.csv"
data_path = "experiment/t04_addition/mod_sum_length_20.csv"

df = read_csv(data_path)


##########
# Debug

def print_row_in_grid(df, row_index):
    # Extract the row and convert it to a list of its values
    row = df.iloc[row_index].tolist()

    # Include headers by prepending the column names to the row data
    headers = df.columns.tolist()
    columns = [headers] + list(zip(*row))

    # Determine the maximum length of string in each column for proper alignment, considering both headers and row data
    max_lengths = [max(len(str(item)) for item in col) for col in zip(*columns)]

    # Print headers and row content with proper alignment
    for col in columns:
        print(" | ".join(str(item).ljust(max_len) for item, max_len in zip(col, max_lengths)))

# Assuming 'df' is your DataFrame and you want to print the first row along with headers in a grid
print_row_in_grid(df, 0)
