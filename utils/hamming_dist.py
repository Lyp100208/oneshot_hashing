#encoding=utf-8
import numpy

def hamming_dist(array_a,array_b):
	if array_a.shape!=array_b.shape:
		print "array_a's shape must be equal to array_b's"
		return 
	else:
		# {0,1}
		dist=0
		for i in range(len(array_a)):
			if array_a[i]!=array_b[i]:
				dist=dist+1
		return dist
if __name__=="__main__":
	a=[0,0,1,1,0,1,0,1,1]
	a=numpy.array(a)
	b=[1,1,0,1,0,0,1,0,1]
	b=numpy.array(b)
	print hamming_dist(a,b)