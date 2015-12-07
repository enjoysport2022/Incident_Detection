#coding=utf-8
import MySQLdb
from sklearn import preprocessing
import numpy as np
conn= MySQLdb.connect(
        host='localhost',
        port = 3306,
        user='root',
        passwd='tm152646',
        db ='teman_db',
        )
cur = conn.cursor()

#创建数据表
# cur.execute("create table student(id int ,name varchar(20),class varchar(30),age varchar(10))")
# cur.execute("create table label(Label VARCHAR(10))")

#插入一条数据
# sqli="insert into student values(%s,%s,%s,%s)"
# cur.execute(sqli,('3','Huhu','2 year 1 class','7'))

#一次插入多条记录
# sqli="insert into student values(%s,%s,%s,%s)"
# cur.executemany(sqli,[
#     ('3','Tom','1 year 1 class','6'),
#     ('3','Jack','2 year 1 class','7'),
#     ('3','Yaheng','2 year 2 class','7'),
#     ])

#修改查询条件的数据
# cur.execute("update student set class='3 year 1 class' where name = 'Tom'")

#删除查询条件的数据
# cur.execute("delete from student where age='9'")

#############################################################################
# 读取数据库中的标签表:
# 查询表中有多少条数据
aa=cur.execute("select * from label")
# print aa

# bb=cur.fetchone()
# print(bb)

label=[]
info = cur.fetchmany(aa)
for ii in info:
    label.append(ii[0])
    # print ii[0]

label=np.array(label)
# print(label.shape)
label=label[1:len(label)]
# print(label.shape)
# print(label)
lb=preprocessing.LabelBinarizer()
dummyY=lb.fit_transform(label)
# dummyY=np.array(dummyY)
# print(dummyY)
# print(len(dummyY))
print("dummyY:------------------------------------------->")
print(dummyY.shape)
print("dummyY:"+str(dummyY))




featureList=[]
featureList_time=[]

###############################################################################
# 读取数据库中的上游表:
aa=cur.execute("select * from up")
# print aa


info = cur.fetchmany(aa)
for ii in info:
    # print ii
    featureList_temp=[]
    for m in range(len(ii)):
        if m==0:
            featureList_time.append(ii[m])
        else:
            featureList_temp.append(float(ii[m]))
            # print(featureList_temp)
    featureList.append(featureList_temp)

featureList=np.array(featureList)
featureList_time=np.array(featureList_time)

# print(featureList_time.shape)
# print(featureList_time)
# print(featureList.shape)
# print(featureList)


###############################################################################
# 读取数据库中的下游表:
aa=cur.execute("select * from down")
# print aa

down=[]
info = cur.fetchmany(aa)
for ii in info:
    down_temp=[]
    # down.append(ii)
    # print ii
    for m in range(1,len(ii)):
        down_temp.append(float(ii[m]))
    down.append(down_temp)
# print(down)

down=np.array(down)
# print(down.shape)
# print(down)
featureList=featureList-down
###########################################################################
featureList=featureList[1:len(featureList)]

print("featureList:------------------------------------------->")

print(featureList.shape)
print(featureList)

# down=down[1:len(down)]
#
# print("down:---------------------------------------------->")
# print(down.shape)
# print(down)
# print(down[0][0])

#########################################################################


cur.close()
conn.commit()
conn.close()

