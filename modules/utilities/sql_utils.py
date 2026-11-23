import sqlite3

"""
If no rule insert(...,rule = None, ...)
"""

class automatic_reports_analysis_db():
        def __init__(self, str_db_name, path):
            self.str_db_name = str_db_name
            self.db_path = path
            self.con = sqlite3.connect(self.db_path)
            self.list_of_tested_module = []

        def __str__(self) -> str:                  
             return f"Database {self.str_db_name} "

        def Create_table_reports_inventory(self):
              """
              Path_reports : directory that contains multiple reports folder
              """
              cursor = self.con.cursor()
              statement = """CREATE TABLE IF NOT EXISTS reports_inventory ( 
              signal_id INTEGER PRIMARY KEY,
              tested_module TEXT NOT NULL,
              campaign TEXT NOT NULL,
              signal_name TEXT NOT NULL,
              rule INTEGER,
              signal_type BOOLEAN NOT NULL,
              equality_state BOOLEAN NOT NULL,
              path TEXT NOT NULL
              )"""
              cursor.execute(statement)

        def Create_table_discrepanies(self):
              """
              If not already there
              """
              cursor = self.con.cursor()
              statement = """CREATE TABLE IF NOT EXISTS discrepancies  ( 
              signal_id INTEGER NOT NULL,
              discrepancy_id INTEGER PRIMARY KEY,
              start_time NUMERIC,
              end_time NUMERIC,
              discrepancy_type BOOLEAN NOT NULL,
              label INTEGER
              )"""
              cursor.execute(statement)

        def create_table_dataset(self, dataset_name):
              """
              If not already there: 
              """
              cursor = self.con.cursor()
              statement = f"""CREATE TABLE IF NOT EXISTS {dataset_name} ( 
              discrepancy_id INTEGER NOT NULL
              )"""
              cursor.execute(statement)
        
        def delete_duplication_report(self):
            print(self)
            cursor = self.con.cursor()
            select =    f"""DELETE FROM reports_inventory
                            WHERE  signal_id NOT IN    (SELECT MIN(signal_id)
                                                FROM   reports_inventory
                                                GROUP  BY   signal_name,
                                                            campaign,
                                                            rule
                                                HAVING MIN(signal_id) IS NOT NULL
                                                )
                        """
            cursor.execute(select)
            self.con.commit()
        def delete_duplication_discrepancies(self):
            print(self)
            cursor = self.con.cursor()
            select = f"""   DELETE FROM discrepancies
                            WHERE  discrepancy_id NOT IN    (SELECT MIN(discrepancy_id)
                                                FROM   discrepancies
                                                GROUP  BY   signal_id,
                                                            start_time,
                                                            end_time
                                                HAVING MIN(discrepancy_id) IS NOT NULL
                                                )
                        """
            cursor.execute(select)
            self.con.commit()

        def insert_reports_inventory(self, tested_module, campaign, signal_name, rule, equality_state, path, print_bool = False):
            if tested_module not in self.list_of_tested_module:
                 self.list_of_tested_module.append(tested_module)
            cursor = self.con.cursor()
            insert_statement = f"""INSERT INTO reports_inventory(tested_module, campaign, signal_name, rule, signal_type, equality_state, path)
            VALUES(?,?,?,?,?,?,?)"""
            str_AorB = signal_name.split("_")
            str_AorB = str_AorB[-1][0]
            if str_AorB == 'A':
                signal_type = 1
            else :
                signal_type =0

            if print_bool == True:
                print("Insert : ", signal_name)

            if rule is None:
                insert_statement = f"""INSERT INTO reports_inventory(tested_module, campaign, signal_name, signal_type, equality_state, path)
                VALUES(?,?,?,?,?,?)"""
                cursor.execute(insert_statement, (tested_module, campaign, signal_name, signal_type, equality_state, path))
            else:
                cursor.execute(insert_statement, (tested_module, campaign, signal_name, rule, signal_type, equality_state, path))
            self.con.commit()

        def insert_discrepancies(self, signal_id, start_time, end_time, discrepancy_type, label = None, print_bool = False):
            cursor = self.con.cursor()
            if label is None:
                insert_statement = f"""INSERT INTO discrepancies(signal_id, start_time, end_time, discrepancy_type)
                                    VALUES(?,?,?,?)"""
                cursor.execute(insert_statement, (signal_id, start_time, end_time, discrepancy_type))

            else:
                insert_statement = f"""INSERT INTO discrepancies(signal_id, start_time, end_time, discrepancy_type, label)
                                    VALUES(?,?,?,?,?)"""
                cursor.execute(insert_statement, (signal_id, start_time, end_time, discrepancy_type, label))
            self.con.commit()
            if print_bool == True:
                print("discrepancy insertion")

        def insert_time_series(self, dataset_name, discrepancy_id):
            """time series is a tuple of tuples"""
            cursor = self.con.cursor()
            insert_statement = f"""INSERT INTO {dataset_name}(discrepancy_id)
                                VALUES(?)"""

            cursor.execute(insert_statement, (discrepancy_id,))
            self.con.commit()
        
        def get_table(self, table_name_str, tested_module = None):
            print(self)
            cursor = self.con.cursor()
            if tested_module is None:
                visualise_statement = f"SELECT * FROM {table_name_str}"
            else:
                visualise_statement = f"SELECT * FROM {table_name_str} WHERE tested_module='{tested_module}'"
            data = cursor.execute(visualise_statement)            
            return data
        
        def get_discrepancies_map(self, tuples = None, number_of_selection = None):
            """
            Return : Joint table of reports and discrepancies : discrepancies, signal_name, path cmp...
            """
            cursor = self.con.cursor()
            join_statement = f"""SELECT discrepancy_id, 
                                        reports_inventory.signal_id, 
                                        path, 
                                        signal_name, 
                                        start_time, 
                                        end_time,
                                        label 
                                FROM reports_inventory
                                INNER JOIN discrepancies ON reports_inventory.signal_id = discrepancies.signal_id"""
            if not tuples is None:
                if len(tuples) == 1:
                    join_statement += f""" WHERE discrepancy_id = '{tuples}'"""
                    data = cursor.execute(join_statement)
                    return data

                if len(tuples) > 1:
                    join_statement += f""" WHERE discrepancy_id IN {tuples}"""
                    if not number_of_selection is None:
                        join_statement += f""" ORDER BY RANDOM() LIMIT {number_of_selection}"""
                    data = cursor.execute(join_statement)
                    return data
                
            return []


        def get_tested_modules(self):
            """
            return: list of all the module present in a report_table:
            """
            cursor = self.con.cursor()
            select_statement = f"""SELECT DISTINCT tested_module FROM reports_inventory"""
            data = cursor.execute(select_statement)
            liste_str = []
            for row in data:
                liste_str.append(row[0])
            return liste_str
        
        def get_signal_ids_by_rule(self, campaign, signal_rule):
            """
            Return: list of indexes using campaign_and_rule
            """
            cursor = self.con.cursor()
            if signal_rule is None:
                select_statement = f"""SELECT signal_id FROM reports_inventory WHERE campaign = '{campaign}' AND rule IS NULL"""
            else:
                select_statement = f"""SELECT signal_id FROM reports_inventory WHERE campaign = '{campaign}' AND rule = '{signal_rule}'"""
            data = cursor.execute(select_statement)
            liste_int = []
            for row in data:
                liste_int.append(row[0])
            return liste_int
        
        def get_discrepancy_ids_by_rule(self, campaign, signal_rule, number_of_selection = None):
            """
            retur: list of indexes using signal_idx
            """
            cursor = self.con.cursor()
            if signal_rule is None:
                join_statement = f"""SELECT discrepancy_id, path, signal_name FROM reports_inventory
                INNER JOIN discrepancies ON reports_inventory.signal_id = discrepancies.signal_id
                WHERE reports_inventory.campaign = '{campaign}' AND reports_inventory.rule IS NULL"""

            else:
                join_statement = f"""SELECT discrepancy_id, path, signal_name FROM reports_inventory
                INNER JOIN discrepancies ON reports_inventory.signal_id = discrepancies.signal_id 
                WHERE reports_inventory.campaign = '{campaign}' AND reports_inventory.rule = '{signal_rule}'"""
            if not number_of_selection is None:
                join_statement = join_statement + f""" ORDER BY RANDOM() LIMIT {number_of_selection}"""
            data = cursor.execute(join_statement)
            liste_int = []
            for row in data:
                liste_int.append(row[0])
            return liste_int
        
        def get_discrepancy_ids_by_rule_2(self, campaign, signal_rule, signal_type = None):
            """
            To elimnate duplication without a complexe algorithme : select one discrepancy from a group of discrancies that have similar start and end time;
            return: list of indexes using signal_idx
            """
            cursor = self.con.cursor()
            if signal_rule is None:
                join_statement = f"""SELECT min(discrepancy_id), start_time, end_time 
                FROM reports_inventory
                INNER JOIN discrepancies
                ON reports_inventory.signal_id = discrepancies.signal_id 
                WHERE reports_inventory.campaign = '{campaign}' AND reports_inventory.rule IS NULL"""
            else:
                join_statement = f"""SELECT min(discrepancy_id), start_time, end_time 
                FROM reports_inventory
                INNER JOIN discrepancies
                ON reports_inventory.signal_id = discrepancies.signal_id 
                WHERE reports_inventory.campaign = '{campaign}' AND reports_inventory.rule = '{signal_rule}'"""
            if not signal_type is None:
                join_statement += f""" AND reports_inventory.signal_type = '{signal_type}'"""

            join_statement += f""" GROUP BY discrepancies.start_time, discrepancies.end_time ORDER BY discrepancies.discrepancy_id"""

            data = cursor.execute(join_statement)
            liste_int = []
            for row in data:
                liste_int.append(row[0])
            return liste_int
            
        
        def get_different_rules(self, tested_module = None):
            """
            return: list of all the module present in a report_table:
            """
            cursor = self.con.cursor()
            select_statement = f"""SELECT DISTINCT tested_module, campaign, rule FROM reports_inventory"""
            if not tested_module is None:
                select_statement =  f"""SELECT DISTINCT tested_module, campaign, rule FROM reports_inventory 
                                        WHERE tested_module = '{tested_module}'"""
            data = cursor.execute(select_statement)
            liste_str = []
            for row in data:
                liste_str.append(row)
            return liste_str
        

        def delete_tested_module(self, tested_module_str):
            cursor = self.con.cursor()
            delete_statement = f"DELETE FROM reports_inventory WHERE tested_module = '{tested_module_str}'"
            cursor.execute(delete_statement)
            self.con.commit()
            self.list_of_tested_module.remove(tested_module_str)
            print(self)

        def delete_table(self, table_name):
            cursor = self.con.cursor()
            delete_statement = f"DELETE FROM {table_name}"
            cursor.execute(delete_statement)
            self.con.commit()
            print(f"Delete table -> {table_name}")
        
        def delete_line_from_table(self, table_name, discrepancy_id):
            cursor = self.con.cursor()
            delete_statement = f"DELETE FROM {table_name} WHERE discrepancy_id = {discrepancy_id}"
            cursor.execute(delete_statement)
            self.con.commit()

        def change_table_name(self, old_name, new_name):
            cursor = self.con.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (old_name,))
            if cursor.fetchone():
                statement = f"ALTER TABLE {old_name} RENAME TO {new_name}"
                cursor.execute(statement)
                self.con.commit()
                print("name of table changed")
        
        def len_table(self, table_name):
            cursor = self.con.cursor()
            statement = f"SELECT COUNT(*) FROM {table_name};"
            cursor.execute(statement)
            nb_rows = cursor.fetchone()[0]
            return nb_rows

        def close(self):
            self.con.commit()
            self.con.close()