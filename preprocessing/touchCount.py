import numpy as np
def checkDup(arr1,arr2):
	count=0
	arrPre=arr1.copy()
	arrCur=arr2.copy()
	while True:
		bDup=False
		for i in range(len(arrCur)):
			if (arrCur[i] in arrPre):
				arrCur[i]=-2
				bDup=True
			if (arrCur[i]!=-2 and arrCur[i]-1 in arrPre):
				arrPre=np.concatenate((arrPre,np.array([arrCur[i]])))
				arrCur[i]=-2
				bDup=True
			if (arrCur[i]!=-2 and arrCur[i]+1 in arrPre):
				arrPre=np.concatenate((arrPre,np.array([arrCur[i]])))
				arrCur[i]=-2
				bDup=True
		if bDup==False:
			break
	for i in range(len(arrCur)):
		if arrCur[i]!=-2 and ((i>0 and (arrCur[i-1]==-2 or np.abs(arrCur[i-1]-arrCur[i])!=1)) or i==0):
			count+=1
	return count

def checkDupIndexes(arr1,arr2):
	count=0
	dupIndex=[]
	arrPre=arr1.copy()
	arrCur=arr2.copy()
	while True:
		bDup=False
		for i in range(len(arrCur)):
			if (arrCur[i] in arrPre):
				arrCur[i]=-2
				bDup=True
			if (arrCur[i]!=-2 and arrCur[i]-1 in arrPre):
				arrPre=np.concatenate((arrPre,np.array([arrCur[i]])))
				arrCur[i]=-2
				bDup=True
			if (arrCur[i]!=-2 and arrCur[i]+1 in arrPre):
				arrPre=np.concatenate((arrPre,np.array([arrCur[i]])))
				arrCur[i]=-2
				bDup=True
		if bDup==False:
			break
	for i in range(len(arrCur)):
		if arrCur[i]!=-2 and ((i>0 and (arrCur[i-1]==-2 or np.abs(arrCur[i-1]-arrCur[i])!=1)) or i==0):
			dupIndex.append(arr2.copy()[i])
	return dupIndex

# Return 4 touchCount values of item
def touchCount(item):
	#Direction: left to right
	countlr=0
	for i in range(item.shape[1]):
		if i==0:
			arr1=np.linspace(0,0,0)
		else:
			arr1=np.where(item[:,i-1]==0)[0]
		arr2=np.where(item[:,i]==0)[0]
		countlr+=checkDup(arr1,arr2)
	#Direction: right to left
	countrl=0
	for i in range(item.shape[1]):
		if i==0:
			arr1=np.linspace(0,0,0)
		else:
			arr1=np.where(item[:,item.shape[1]-i]==0)[0]
		arr2=np.where(item[:,item.shape[1]-i-1]==0)[0]
		countrl+=checkDup(arr1,arr2)
	#Direction: top down
	counttd=0
	for i in range(item.shape[0]):
		if i==0:
			arr1=np.linspace(0,0,0)
		else:
			arr1=np.where(item[i-1]==0)[0]
		arr2=np.where(item[i]==0)[0]
		counttd+=checkDup(arr1,arr2)
	#Direction: bottom up
	countbu=0
	for i in range(item.shape[0]):
		if i==0:
			arr1=np.linspace(0,0,0)
		else:
			arr1=np.where(item[item.shape[0]-i]==0)[0]
		arr2=np.where(item[item.shape[0]-i-1]==0)[0]
		countbu+=checkDup(arr1,arr2)
	return countlr, countrl, counttd, countbu

# Return standard tensor for model input - shape (68,64)
def touchCountTensor(item):
	countlr, countrl, counttd, countbu = touchCount(item) 
	result=np.concatenate([ np.concatenate([ np.linspace(1,1,countlr), np.linspace(0,0,64-countlr) ]), np.concatenate([ np.linspace(1,1,countrl), np.linspace(0,0,64-countrl) ]), np.concatenate([ np.linspace(1,1,counttd), np.linspace(0,0,64-counttd) ]), np.concatenate([ np.linspace(1,1,countbu), np.linspace(0,0,64-countbu) ]) ])
	result=result.reshape(4,64)
	retVal=np.concatenate([item[0:item.shape[0]],result])
	return retVal

# Return indexes of touchPoint 
def touchIndexTensor(item):
	#Direction: left to right
	indexlr=[]
	for i in range(item.shape[1]):
		if i==0:
			arr1=np.linspace(0,0,0)
		else:
			arr1=np.where(item[:,i-1]==0)[0]
		arr2=np.where(item[:,i]==0)[0]
		indexlr.extend(checkDupIndexes(arr1,arr2))
	#Direction: right to left
	indexrl=[]
	for i in range(item.shape[1]):
		if i==0:
			arr1=np.linspace(0,0,0)
		else:
			arr1=np.where(item[:,item.shape[1]-i]==0)[0]
		arr2=np.where(item[:,item.shape[1]-i-1]==0)[0]
		indexrl.extend(checkDupIndexes(arr1,arr2))
	#Direction: top down
	indextd=[]
	for i in range(item.shape[0]):
		if i==0:
			arr1=np.linspace(0,0,0)
		else:
			arr1=np.where(item[i-1]==0)[0]
		arr2=np.where(item[i]==0)[0]
		indextd.extend(checkDupIndexes(arr1,arr2))
	#Direction: bottom up
	indexbu=[]
	for i in range(item.shape[0]):
		if i==0:
			arr1=np.linspace(0,0,0)
		else:
			arr1=np.where(item[item.shape[0]-i]==0)[0]
		arr2=np.where(item[item.shape[0]-i-1]==0)[0]
		indexbu.extend(checkDupIndexes(arr1,arr2))
	result=np.concatenate([ 
		np.array([1. if i in indexlr else 0 for i in range(64)]), 
		np.array([1. if i in indexrl else 0 for i in range(64)]),
		np.array([1. if i in indextd else 0 for i in range(64)]),
		np.array([1. if i in indexbu else 0 for i in range(64)])])
	result=result.reshape(4,64)
	retVal=np.concatenate([item,result])
	return retVal

# Return indexes of touchPoint 
def touchIndexTensor2(item):
	#Direction: left to right
	indexlr=[]
	for i in range(item.shape[1]):
		if i==0:
			arr1=np.linspace(0,0,0)
		else:
			arr1=np.where(item[:,i-1]==0)[0]
		arr2=np.where(item[:,i]==0)[0]
		indexlr.extend(checkDupIndexes(arr1,arr2))
	#Direction: right to left
	indexrl=[]
	for i in range(item.shape[1]):
		if i==0:
			arr1=np.linspace(0,0,0)
		else:
			arr1=np.where(item[:,item.shape[1]-i]==0)[0]
		arr2=np.where(item[:,item.shape[1]-i-1]==0)[0]
		indexrl.extend(checkDupIndexes(arr1,arr2))
	#Direction: top down
	indextd=[]
	for i in range(item.shape[0]):
		if i==0:
			arr1=np.linspace(0,0,0)
		else:
			arr1=np.where(item[i-1]==0)[0]
		arr2=np.where(item[i]==0)[0]
		indextd.extend(checkDupIndexes(arr1,arr2))
	#Direction: bottom up
	indexbu=[]
	for i in range(item.shape[0]):
		if i==0:
			arr1=np.linspace(0,0,0)
		else:
			arr1=np.where(item[item.shape[0]-i]==0)[0]
		arr2=np.where(item[item.shape[0]-i-1]==0)[0]
		indexbu.extend(checkDupIndexes(arr1,arr2))
	result=np.concatenate([ 
		np.repeat( np.linspace(0,0,64),4),
		np.repeat( np.array([1. if i in indexlr else 0 for i in range(64)]) ,3), 
		np.repeat( np.array([1. if i in indexrl else 0 for i in range(64)]) ,3),
		np.repeat( np.array([1. if i in indextd else 0 for i in range(64)]) ,3),
		np.repeat( np.array([1. if i in indexbu else 0 for i in range(64)]) ,3)])
	result=result.reshape(16,64)
	retVal=np.concatenate([item,result])
	return retVal