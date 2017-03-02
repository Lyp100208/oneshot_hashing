#encoding=utf-8
import os
import numpy
import hamming_dist

#hash code: list[ [ uint[1,0,1,...0], uint:label],..]
def read_usual(path):
	data_path_label=open(path).readlines()
        #print data_path_label[500]
 	data_dict=dict()
 	for i in range(80):
 	    for j in range(500):
	 	if not data_dict.has_key(i):
	 	    data_dict[i]=list()
	 	    data_dict[i].append(data_path_label[i*500+j])
	 	else:
	 	    data_dict[i].append(data_path_label[i*500+j])
	#print len(data_dict)
	#print data_dict[0]

 	return data_dict	

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


def calculate_ap(query,database):# 
	database_capacity=len(database)
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


def calculate_map_per_class(query_list,database,query_class):
	query_capacity=len(query_list)
	query_per_class=query_capacity/query_class

	map_per_class_list=list()
	for i in range(query_class):
		map_per_class=float(0)
		for j in range(query_per_class):
			ap,related_count=calculate_ap(query_list[i*j+j],database)
			map_per_class += ap
		map_per_class = map_per_class/query_per_class
		map_per_class_list.append(map_per_class)

	return map_per_class_list,sum(map_per_class_list)/(query_class)


def create_filtered_usual(map_per_class_list,usual_data_dict):

	temp=list(numpy.array(map_per_class_list).argsort())
	sorted_keys=list(reversed(temp))[0:40]
        print sorted_keys

	filtered_usual=list()
	for i in range(len(sorted_keys)):
            #print usual_data_dict[sorted_keys[i]][0:20]
	    filtered_usual.extend(usual_data_dict[sorted_keys[i]])


	f=open("./data/cifar100/filtered_usual.txt","w+")
	f.writelines(filtered_usual)
	f.close()
    


if __name__=="__main__":
	os.chdir("/home/libing/dl/oneshot_hashing/")
	usual_data_dict=read_usual("./data/cifar100/train_usual.txt")
	query_list=read_query_list(["./feature/cifar100/googlenet_usual_48/val_feature.txt","./feature/cifar100/googlenet_usual_48/val_label.txt"])             
	db_list=read_db_list(["./feature/cifar100/googlenet_usual_48/train_feature.txt","./feature/cifar100/googlenet_usual_48/train_label.txt","./feature/cifar100/googlenet_usual_48/test_feature.txt","./feature/cifar100/googlenet_usual_48/test_label.txt"])

	map_per_class_list,MAP=calculate_map_per_class(query_list,db_list,80)
	print map_per_class_list
	print MAP
	
        for i in range(len(map_per_class_list)):
            map_per_class_list[i]=str(map_per_class_list[i])+"\n"

	f=open("./data/cifar100/map_per_class.txt","w+")
        f.writelines(map_per_class_list)
        f.close()

        create_filtered_usual(map_per_class_list,usual_data_dict)
        
