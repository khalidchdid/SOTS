import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


def count_per_class(db_path: str) -> pd.DataFrame:
    """
    Count the number of examples per class in the discrepancies table.

    Args:
        db_path : path to the .db SQLite database file
        table   : name of the table (default: "discrepancies")
    Returns:
        pd.DataFrame with columns ["class", "count"]
    """
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT label, COUNT(*) as count
        FROM discrepancies
        GROUP BY label
        ORDER BY count DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Plot the class distribution as a bar chart.

    Args:
        df : output of count_per_class()
    """
    plt.figure(figsize=(10, 5))
    plt.bar(df["label"].astype(str), df["count"], color="steelblue", alpha=0.85)
    plt.xlabel("Class")
    plt.ylabel("Number of examples")
    plt.title("Class distribution in the discrepancies table")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# Usage
db_path = "/workspace/kakhmouch/data/database_analog_F1.db"
df = count_per_class(db_path)
print(df.to_string(index=False))

"""
Results :

ANALOG                  
 label  count                
     7   5964
     3   4704
     2   1935
     4    772
     1    717
    11    417
     6    257
     8    166
    10     84

BINARY
 label  count
     7   1532
     4   1131
     9    986
     8    499
     6    437
     2    354
     3     64
     1     32
    10      8
    11      4

ALL
 label  count
     7   7496
     3   4768
     2   2289
     4   1903
     9    986
     1    749
     6    694
     8    665
    11    421
    10     92

"""
