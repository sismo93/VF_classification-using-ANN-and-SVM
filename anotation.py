
from data_structures import Annotation
import numpy as np





def createAnnotationArray(indexArray,labelArray,hi,NSRsymbol):

	annotations = []

	for i in range(len(indexArray)):

		annotations.append(Annotation(index=indexArray[i],label=labelArray[i]))

	distributedAnnotations = createDistributedAnnotations(annotationArray=annotations,hi=hi,NSRsymbol=NSRsymbol)

	return distributedAnnotations


def createDistributedAnnotations(annotationArray,hi,NSRsymbol):

	labelArray=[]

	localLo = 0
	localHi = annotationArray[0].index
	currLabel = NSRsymbol

	## The following is similar to interval covering algorithms

	## We are assuming the first unannotated part to be NSR

	for i in range(localLo,localHi):

		labelArray.append(currLabel)


	## now for the other actual annotated segments

	for i in range(1,len(annotationArray)):			
													# interval
		localLo = annotationArray[i-1].index
		localHi = annotationArray[i].index
		currLabel = annotationArray[i-1].label

		for j in range(localLo,localHi):

			labelArray.append(currLabel)

	## for the last segment

	localLo = annotationArray[len(annotationArray)-1].index
	localHi = hi
	currLabel = annotationArray[len(annotationArray)-1].label

	for j in range(localLo, localHi):
		labelArray.append(currLabel)

	return labelArray				# point wise annotation array







def createCUDBAnnotation(annotationIndex,annotationArr,lenn):


	li = []							# initialize

	for i in range(lenn):						
		li.append('notVF')					

	st=-1
	en=-1


	for i in range(len(annotationArr)):

		if(annotationArr[i]=='N'):				

			li[i]='NSR'

	for i in range(len(annotationArr)):

		if(annotationArr[i]=='['):

			st = annotationIndex[i]

		if(annotationArr[i]==']'):

			en = annotationIndex[i]

			for j in range(st,en+1):

				li[j]='VF'
			st = -1
			en = -1

	if(st!=-1):

		for j in range(st,lenn):
			li[j] = 'VF'

	return np.array(li)