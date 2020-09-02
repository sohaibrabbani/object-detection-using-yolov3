import mysql.connector


class DatabasePipeline(object):

    def __init__(self):
        self.create_connection()

    def create_connection(self):
        self.conn = mysql.connector.connect(
            host='localhost',
            user='root',
            passwd='password',
            database='objectdetectiondb'
        )
        self.cursor = self.conn.cursor()

    def store_db(self, items):
        try:
            for item in items:
                self.cursor.execute("""insert into intruders (`time`,image_path,video_path) values (%s, %s, %s)""", (
                    item[0],
                    item[1],
                    item[2]
                ))
                self.conn.commit()
            print(self.cursor.rowcount, "Record inserted successfully into Laptop table")
            self.cursor.close()

        except mysql.connector.Error as error:
            print("Failed to insert record into Laptop table {}".format(error))

        finally:
            if self.conn.is_connected():
                self.conn.close()
                print("MySQL connection is closed")
