import mysql.connector
from mysql.connector import Error

def get_pelanggar_by_label(label):
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='django_cnn'
        )
        if conn.is_connected():
            cursor = conn.cursor(dictionary=True)
            query = "SELECT nama, umur, kasus, tim, foto FROM blacklist WHERE label = %s"
            cursor.execute(query, (label,))
            result = cursor.fetchone()
            return result
    except Error as e:
        print("Database error:", e)
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
