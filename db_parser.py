import struct
import sqlite3

if __name__ == "__main__":
    db_file = "src/common/data/gold_std/data.db"
    conn = sqlite3.connect(db_file)
    sql_query = """SELECT name FROM sqlite_master
    WHERE type='table';"""
 
    # Creating cursor object using connection object
    cursor = conn.cursor()

    # executing our sql query
    cursor.execute(sql_query)
    print("List of tables\n")

    # printing all tables list
    tables = cursor.fetchall()
    for table in tables:
        table = table[0]
        cursor.execute(f"SELECT * FROM {table}")
        print(f"Table {table}: {[description[0] for description in cursor.description]}", flush=True)

    # # get camera intrinsics
    # cursor.execute("SELECT calibration FROM Data WHERE calibration IS NOT NULL")
    # rows = [elem[0] for elem in cursor.fetchall()]
    # formats = ['I', 'H', 'f', 'd']
    # for row in rows:
    #     try:
    #         # decode
    #         calibration_data = struct.unpack('8I', row[:struct.calcsize('8I')])
    #         print(f"Calibration Data: {calibration_data}")
    #     except struct.error as e:
    #         print("Failed to unpack calibration data:", e)
    #     break
