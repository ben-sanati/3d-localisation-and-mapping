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
    print("\n" + "=" * 100 + "\n")

    cursor.execute("SELECT * FROM Node")
    node = cursor.fetchall()
    n1 = node[0]
    print(f"Node ({len(node)}): {n1[:3]}, {n1[-4]},  {n1[-1]}) = (id, map_id, weight, label, time_enter)")

    with open("node.txt", "w") as file:
        for n in node:
            file.write(f"Node ({len(node)}): ({n[0]}, {n[2]}) = (id, weight)\n")

    cursor.execute("SELECT * FROM Link")
    link = cursor.fetchall()
    l1 = link[0]
    print(f"Link ({len(link)}): {l1[:3]} = (id, map_id, weight)")

    with open("link.txt", "w") as file:
        for l in link:
            file.write(f"Link ({len(link)}): {l[:3]} = (id, map_id, weight)\n")

    cursor.execute("SELECT * FROM Feature")
    feature = cursor.fetchall()
    f1 = feature[0]
    print(f"Feature ({len(feature)}): {f1[:-1]} = ('node_id', 'word_id', 'pos_x', 'pos_y', 'size', 'dir', 'response', 'octave', 'depth_x', 'depth_y', 'depth_z', 'descriptor_size')")

    with open("feature.txt", "w") as file:
        for f in feature:
            file.write(f"Feature ({len(feature)}): {f[:-1]} = ('node_id', 'word_id', 'pos_x', 'pos_y', 'size', 'dir', 'response', 'octave', 'depth_x', 'depth_y', 'depth_z', 'descriptor_size')\n")
