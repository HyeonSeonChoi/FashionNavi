import oracledb

con = oracledb.connect(user="system", password="admin", dsn="203.250.123.49/xe")
cursor = con.cursor()

# cursor.execute("insert into style (image_path, upper_feature, lower_feature, upper_color, lower_color) values (:1, :2, :3, :4, :5)", [1, 2, 3, 4, 5])
cursor.execute("drop table style")
cursor.execute("SELECT * FROM style")
out_data = cursor.fetchone()
print("=====>", out_data)
# con.commit()

con.close()
