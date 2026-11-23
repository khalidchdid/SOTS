from pathlib import Path

import sqlite3
import pandas as pd

def write_txt(db_path, txt_file_name):
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)

    p = Path(db_path)
    new_p = p.with_name(txt_file_name)

    with open(new_p, "w", encoding="utf-8") as f:
        for table in tables['name']:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            f.write(f"######{table}#######\n")
            f.write(df.to_string())
            f.write("\n\n")
    conn.close()

if __name__ == "__main__":
    db_path = "/workspace/kakhmouch/data/database_training.db"
    txt_file_name = "database_training.txt"
    write_txt(db_path, txt_file_name)
    print("written")