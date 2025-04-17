import sqlite3
import pandas as pd

class face_data_pipeline:
    def __init__(self, db_name):
        """Initialize with a database name."""
        self.db_name = f"{db_name}.db"
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
    
    def create_table(self, table_name, schema):
        """Create a table with a given schema."""
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        self.cursor.execute(query)
        self.conn.commit()
        print(f"Table '{table_name}' created successfully.")
    
    def insert_data(self, table_name, data,columns):
        
        df = pd.DataFrame(data, columns=columns)


        df.to_sql(f"{table_name}", self.conn, if_exists="append", index=False)
        print("success inserted data")
    
    def add_member(self, table_name, member_data):
        """Add a single new member."""
        columns = ", ".join(member_data.keys())
        placeholders = ", ".join(["?" for _ in member_data])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        
        self.cursor.execute(query, tuple(member_data.values()))
        self.conn.commit()
        print(f"Added new member: {member_data['name']}")
    
    def update_member(self, table_name, member_id, updates):
        """Update specific fields for a given member."""
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [member_id]
        query = f"UPDATE {table_name} SET {set_clause} WHERE id = ?"
        
        self.cursor.execute(query, values)
        self.conn.commit()
        print(f"Updated member ID {member_id} in {table_name}.")
    
    def view_all(self, table_name):
        """Retrieve all records from a table."""
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, self.conn)
        return df
    
    def view_filtered(self, table_name, field, value):
        """Retrieve records based on a specific condition."""
        query = f"SELECT * FROM {table_name} WHERE {field} = ?"
        df = pd.read_sql(query, self.conn, params=(value,))
        return df
    
    def max_id(self, table_name):
        """Retrieve all records from a table."""
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, self.conn)
        print(f" you must enter an id greater then {max(df['id'])}")
        return max(df['id'])
    def delete_member(self, table_name, member_id):
        """Delete a member from the table by ID."""
        query = f"DELETE FROM {table_name} WHERE id = ?"
        self.cursor.execute(query, (member_id,))
        self.conn.commit()
        print(f"Deleted member with ID {member_id} from '{table_name}'.")

    def name_exists(self, table_name, name):
        """Check if a name already exists in the table (case-insensitive)."""
        query = f"SELECT COUNT(*) FROM {table_name} WHERE LOWER(name) = LOWER(?)"
        self.cursor.execute(query, (name,))
        count = self.cursor.fetchone()[0]
        return count > 0
    def close_connection(self):
        """Close the database connection."""
        self.conn.close()
        print("Database connection closed.")