#encoding=utf-8
import os
import numpy
import hamming_dist

#hash code: list[ [ uint[1,0,1,...0], uint:label],..]

def read_query_list(path):
	feature_path=path[0]
	label_path=path[1]
	f=open(feature_path)
	feature_list=f.readlines()
	f.close()
	f=open(label_path)
	label_list=f.readlines()
	f.close()

	assert len(feature_list)==len(label_list)

	query_list=list()
	for i in range(len(feature_list)):
		feature=feature_list[i].strip().split()
		feature=map(float,feature)
		#print feature
		#print "\n"
		for j in range(len(feature)):
			if feature[j]>=0.5:
				feature[j]=int(1)
			else:
				feature[j]=int(0)
		#print feature
		query_hashcode=feature
		query_label=int(label_list[i].strip())
		query_list.append([query_hashcode,query_label])

	return query_list

def read_db_list(path):
        
	train_feature_path=path[0]
	train_label_path=path[1]
	f=open(train_feature_path)
	train_feature_list=f.readlines()
	f.close()
	f=open(train_label_path)
	train_label_list=f.readlines()
	f.close()
        if len(path) == 4:
	    test_feature_path=path[2]
	    test_label_path=path[3]
	    f=open(test_feature_path)
    	    test_feature_list=f.readlines()
	    f.close()
	    f=open(test_label_path)
	    test_label_list=f.readlines()
	    f.close()
        if len(path) < 4:
            test_feature_list=list()
            test_label_list=list()

	feature_list=train_feature_list+test_feature_list
	label_list=train_label_list+test_label_list

	assert len(feature_list)==len(label_list)

	db_list=list()
	for i in range(len(feature_list)):
		feature=feature_list[i].strip().split()
		feature=map(float,feature)
		#print feature
		#print "\n"
		for j in range(len(feature)):
			if feature[j]>=0.5:
				feature[j]=int(1)
			else:
				feature[j]=int(0)
		#print feature
		db_hashcode=feature
		db_label=int(label_list[i].strip())
		db_list.append([db_hashcode,db_label])
		
	return db_list


def calculate_ap(query,database,retrieval_capacity):# 
	database_capacity=len(database)
        print "database_capacity:"+str(database_capacity)
	dist_list=list()
	query_hashcode=query[0]
	query_label=query[1]
	#print query_label
	for i in range(database_capacity):
		#query,database[i]
		db_hashcode=database[i][0]
		db_label=database[i][1]
		dist=hamming_dist.hamming_dist(numpy.array(query_hashcode),numpy.array(db_hashcode))
		dist_list.append([dist,db_label])# int ,int
	
	dist_list.sort(key=lambda x :x[0])
	#retrieval_list=dist_list[0:retrieval_capacity]
        retrieval_list=dist_list
	#print retrieval_list
	average_precision=float(0)
	related_count=float(0)

	for i in range(database_capacity):
		if retrieval_list[i][1]==query_label:
			related_count += 1
			rank=float(i+1)
			average_precision += float(related_count/rank)

	#average_precision = average_precision/float(retrieval_capacity)
        average_precision=average_precision/float(related_count)
	return average_precision,related_count


def calculate_map(query_list,database,retrieval_capacity):
	query_capacity=len(query_list)
	sum_ap=float(0)
	for i in range(query_capacity):
		ap,related_count=calculate_ap(query_list[i],database,retrieval_capacity)
		print related_count
                print ap
                print "\n"
		sum_ap += ap
	map = sum_ap/float(query_capacity)
	return map

if __name__=="__main__":
	
	threshold=float(0.5)
	retrieval_capacity=200
	
	query_list=read_query_list(["./feature/sun397/googlenet_usual43_and_learning_48/val_feature.txt","./feature/sun397/googlenet_usual43_and_learning_48/val_label.txt"])
                
	#print query_list
	db_list=read_db_list(["./feature/sun397/googlenet_usual43_and_learning_48/train_feature.txt","./feature/sun397/googlenet_usual43_and_learning_48/train_label.txt","./feature/sun397/googlenet_usual43_and_learning_48/test_feature.txt","./feature/sun397/googlenet_usual43_and_learning_48/test_label.txt"])
	#print query_list
	#print db_list

	map=calculate_map(query_list,db_list,retrieval_capacity)
	print "map: "+str(map)
        f=open("./utils/map.log","aw+")
        f.write("sun397 iter=120000,stepsize=30000,learning_rate=0.001,  googlenet_usual43_and_learning_48 :对usual43: map: "+str(map)+"\n")
        f.close()




