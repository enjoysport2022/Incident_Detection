__author__ = 'haorui2'
import MySQLdb
try:
    conn=MySQLdb.connect(host='localhost',user='root',passwd='tm152646',db='teman_db',port=3306)
    cur=conn.cursor()
    cur.execute('select * from user_tbl')
    cur.close()
    conn.close()
except MySQLdb.Error,e:
     print "Mysql Error %d: %s" % (e.args[0], e.args[1])