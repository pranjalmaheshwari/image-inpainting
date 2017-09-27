#!/usr/bin/env
#author@Pranjal

import sys
import numpy as np
import logging
import cv2
import cv
import time
from Queue import PriorityQueue
logging.basicConfig(stream=sys.stderr,level=logging.DEBUG)


def printImage(img):
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			print 'src ' + str(img[i][j])

def editBorder(img):
	lstR = img.shape[0] - 1
	lstC = img.shape[1] - 1
	for i in range(img.shape[0]):
		img[i,0] = 0
		img[i,lstC] = 0
	for i in range(img.shape[1]):
		img[0,i] = 0
		img[lstR,i] = 0

def fms(i1,j1,i2,j2,f,t):
	a1 = t[i1,j1]
	a2 = t[i2,j2]
	m = min(a1,a2)
	if(f[i1,j1] != 2):
		if(f[i2,j2] != 2):
			if(abs(a1-a2) >= 1.0):
				ret = 1.0 + m
			else:
				ret = 0.5*(a1 + a2 + np.sqrt(2.0-(a1-a2)*(a1-a2)))
		else:
			ret = 1.0 + a1
	elif(f[i2,j2] != 2):
		ret = 1.0 + a2
	else:
		ret = 1.0 + m
	return ret

def min4(a1,a2,a3,a4):
	a = min(a1,a2)
	a = min(a,a3)
	return min(a,a4)

def inPaintPoint(i,j,f,t,ret,epsilon):
	radiusSqr = float(epsilon*epsilon)

	gradTX = 0.0
	gradTY = 0.0

	if(f[i,j+1] != 2):
		if(f[i,j-1] != 2):
			gradTX = 0.5*(t[i,j+1] - t[i,j-1])
		else:
			gradTX = (t[i,j+1] - t[i,j])
	else:
		if(f[i,j-1] != 2):
			gradTX = (t[i,j] - t[i,j-1])
		# else:
		# 	gradTX = 0.0

	if(f[i+1,j] != 2):
		if(f[i-1,j] != 2):
			gradTY = 0.5*(t[i+1,j] - t[i-1,j])
		else:
			gradTY = (t[i+1,j] - t[i,j])
	else:
		if(f[i-1,j] != 2):
			gradTY = (t[i,j] - t[i-1,j])
		# else:
		# 	gradTY = 0.0

	
	minI = max(1,i - epsilon)
	minJ = max(1,j - epsilon)
	maxI = min(ret.shape[0] - 1,i + epsilon + 1)
	maxJ = min(ret.shape[1] - 1,j + epsilon + 1)

	if(len(ret.shape) == 3):

		Ia = np.array([0.0,0.0,0.0])
		Ix = np.array([0.0,0.0,0.0])
		Iy = np.array([0.0,0.0,0.0])
		s = 1.0e-20
		gradIX = np.array([0.0,0.0,0.0])
		gradIY = np.array([0.0,0.0,0.0])

		for k in range (minI,maxI):
			for l in range(minJ,maxJ):
				rY = float(i - k)
				rX = float(j - l)
				if((f[k,l] != 2) and ((rX*rX + rY*rY) <= radiusSqr)):
					dst = 1.0/((rX*rX + rY*rY)*np.sqrt(rX*rX + rY*rY))
					lev = 1.0/(1.0 + abs(t[k,l] - t[i,j]))
					dirT = ((gradTX * rX) + (gradTY * rY))
					if(abs(dirT)<=0.01):
						dirT=0.000001
					w = abs(dst*lev*dirT)

					if(f[k,l+1] != 2):
						if(f[k,l-1] != 2):
							gradIX = 2.0*ret[k,l+1,:] - ret[k,l-1,:]
						else:
							gradIX = ret[k,l+1,:] - ret[k,l,:]
					else:
						if(f[k,l-1] != 2):
							gradIX = ret[k,l,:] - ret[k,l-1,:]
						# else:
						# 	gradIX = 0.0

					if(f[k+1,l] != 2):
						if(f[k-1,l] != 2):
							gradIY = 2.0*ret[k+1,l,:] - ret[k-1,l,:]
						else:
							gradIY = ret[k+1,l,:] - ret[k,l,:]
					else:
						if(f[k-1,l] != 2):
							gradIY = ret[k,l,:] - ret[k-1,l,:]
						# else:
						# 	gradIY = 0.0

					#print gradIX , gradIY

					Ia += w * ret[k,l,:]
					Ix -= w * rX * gradIX
					Iy -= w * rY * gradIY
					s += w
		ret[i,j,0] = 0.5 + (Ia[0]/s) + (Ix[0]+Iy[0])/(np.sqrt(Ix[0]*Ix[0]+Iy[0]*Iy[0])+1.0e-20)
		ret[i,j,1] = 0.5 + (Ia[1]/s) + (Ix[1]+Iy[1])/(np.sqrt(Ix[1]*Ix[1]+Iy[1]*Iy[1])+1.0e-20)
		ret[i,j,2] = 0.5 + (Ia[2]/s) + (Ix[2]+Iy[2])/(np.sqrt(Ix[2]*Ix[2]+Iy[2]*Iy[2])+1.0e-20)

	else:

		Ia = 0.0
		Ix = 0.0
		Iy = 0.0
		s = 1.0e-20
		gradIX = 0.0
		gradIY = 0.0

		for k in range (minI,maxI):
			for l in range(minJ,maxJ):
				rY = i - k
				rX = j - l
				if((f[k,l] != 2) and ((rX*rX + rY*rY) <= radiusSqr)):
					dst = 1.0/((float(rX*rX + rY*rY))*(np.sqrt(rX*rX + rY*rY)))
					lev = 1.0/(1.0 + abs(t[k,l] - t[i,j]))
					dirT = ((gradTX * float(rX)) + (gradTY * float(rY)))
					if(abs(dirT)<=0.01):
						dirT=0.000001
					w = abs(dst*lev*dirT)

					if(f[k,l+1] != 2):
						if(f[k,l-1] != 2):
							gradIX = 2.0*ret[k,l+1] - ret[k,l-1]
						else:
							gradIX = ret[k,l+1] - ret[k,l]
					else:
						if(f[k,l-1] != 2):
							gradIX = ret[k,l] - ret[k,l-1]
						# else:
						# 	gradIX = 0.0

					if(f[k+1,l] != 2):
						if(f[k-1,l] != 2):
							gradIY = 2.0*ret[k+1,l] - ret[k-1,l]
						else:
							gradIY = ret[k+1,l] - ret[k,l]
					else:
						if(f[k-1,l] != 2):
							gradIY = ret[k,l] - ret[k-1,l]
						# else:
						# 	gradIY = 0.0

					Ia += w * ret[k,l]
					Ix -= w * gradIX * float(rX)
					Iy -= w * gradIY * float(rY)
					s += w
		ret[i,j] = 0.5 + (Ia/s) + (Ix+Iy)/(np.sqrt(Ix*Ix+Iy*Iy)+1.0e-20)
	#print ret[i,j]

def telea(f,t,ret,epsilon,heap):

	while(heap.empty() == False):
		(valT,(r,c)) = heap.get()
		# print heap.qsize()
		f[r,c] = 0
		if(f[r-1,c] == 2):
			i = r-1
			j = c
			dist = min4(fms(i-1,j,i,j-1,f,t),fms(i+1,j,i,j-1,f,t),fms(i-1,j,i,j+1,f,t),fms(i+1,j,i,j+1,f,t))
			t[i,j] = dist
			#do something
			inPaintPoint(i,j,f,t,ret,epsilon)
			f[i,j] = 1
			heap.put((dist,(i,j)))
		if(f[r+1,c] == 2):
			i = r+1
			j = c
			dist = min4(fms(i-1,j,i,j-1,f,t),fms(i+1,j,i,j-1,f,t),fms(i-1,j,i,j+1,f,t),fms(i+1,j,i,j+1,f,t))
			t[i,j] = dist
			#do something
			inPaintPoint(i,j,f,t,ret,epsilon)
			f[i,j] = 1
			heap.put((dist,(i,j)))
		if(f[r,c-1] == 2):
			i = r
			j = c-1
			dist = min4(fms(i-1,j,i,j-1,f,t),fms(i+1,j,i,j-1,f,t),fms(i-1,j,i,j+1,f,t),fms(i+1,j,i,j+1,f,t))
			t[i,j] = dist
			#do something
			inPaintPoint(i,j,f,t,ret,epsilon)
			f[i,j] = 1
			heap.put((dist,(i,j)))
		if(f[r,c+1] == 2):
			i = r
			j = c+1
			dist = min4(fms(i-1,j,i,j-1,f,t),fms(i+1,j,i,j-1,f,t),fms(i-1,j,i,j+1,f,t),fms(i+1,j,i,j+1,f,t))
			t[i,j] = dist
			#do something
			inPaintPoint(i,j,f,t,ret,epsilon)
			f[i,j] = 1
			heap.put((dist,(i,j)))

def inPaint(img,inpaint_mask,epsilon):
	# print 'InPainting Started\n\nPress ^C to exit\n\n'
	padRow = img.shape[0] + 2
	padCol = img.shape[1] + 2

	cross = cv.CreateStructuringElementEx( cols=3, rows=3, anchorX=1, anchorY=1, shape=cv.CV_SHAPE_CROSS )
	rng = cv.CreateStructuringElementEx( 2*epsilon+1, 2*epsilon+1, epsilon, epsilon, shape=cv.CV_SHAPE_RECT )

	if(len(img.shape) == 0):
		ret = np.empty(shape=(padRow,padCol),dtype='uint8')
		ret[1:-1,1:-1] = img
		ret[1:-1,0] = img[:,0]
		ret[1:-1,-1] = img[:,-1]
		ret[0,1:-1] = img[0,:]
		ret[1,1:-1] = img[1,:]
		ret[0,0] = img[0,0]
		ret[-1,0] = img[-1,0]
		ret[0,-1] = img[0,-1]
		ret[-1,-1] = img[-1,-1]
	if(len(img.shape) == 3):
		ret = np.empty(shape=(padRow,padCol,3),dtype='uint8')
		ret[1:-1,1:-1,:] = img[:,:,:]
		ret[1:-1,0,:] = img[:,0,:]
		ret[1:-1,-1,:] = img[:,-1,:]
		ret[0,1:-1,:] = img[0,:,:]
		ret[1,1:-1,:] = img[1,:,:]
		ret[0,0,:] = img[0,0,:]
		ret[-1,0,:] = img[-1,0,:]
		ret[0,-1,:] = img[0,-1,:]
		ret[-1,-1,:] = img[-1,-1,:]

	ret = ret.astype(float)

	# f = np.zeros(shape=(padRow,padCol),dtype='uint8')
	t = np.empty(shape=(padRow,padCol),dtype=float)
	cv.Set(cv.fromarray(t),cv.Scalar(1.0e6,0,0,0))

	mask = np.zeros(shape=(padRow,padCol),dtype='uint8')
	mask[1:-1,1:-1] = 2*(inpaint_mask/255)

	band = np.zeros(shape=(padRow,padCol),dtype='uint8')
	cv.Dilate(cv.fromarray(mask),cv.fromarray(band),cross,1)
	band -= mask
	editBorder(band)
	heap = PriorityQueue()
	heapOut = PriorityQueue()
	# cv.Set(cv.fromarray(f),cv.Scalar(1,0,0,0),cv.fromarray(band))
	# cv.Set(cv.fromarray(f),cv.Scalar(2,0,0,0),cv.fromarray(mask))
	cv.Set(cv.fromarray(t),cv.Scalar(0.0,0,0,0),cv.fromarray(band))
	for (r,c) , val in np.ndenumerate(band):
		if(val == 2):
			heap.put((0.0,(r,c)))
			heapOut.put((0.0,(r,c)))

	out = np.zeros(shape=(padRow,padCol),dtype='uint8')
	cv.Dilate(cv.fromarray(mask),cv.fromarray(out),rng,1)
	out -= mask
	out -= band
	editBorder(out)

	print 'Initialization Completed'

	while(heapOut.empty() == False):
		(valT,(r,c)) = heapOut.get()
		out[r,c] = 3
		if(out[r-1,c] == 2):
			i = r-1
			j = c
			dst = min4(fms(i-1,j,i,j-1,out,t),fms(i+1,j,i,j-1,out,t),fms(i-1,j,i,j+1,out,t),fms(i+1,j,i,j+1,out,t))
			t[i,j] = dst
			out[i,j] = 1
			heapOut.put((dst,(i,j)))
		if(out[r,c-1] == 2):
			i = r
			j = c-1
			dst = min4(fms(i-1,j,i,j-1,out,t),fms(i+1,j,i,j-1,out,t),fms(i-1,j,i,j+1,out,t),fms(i+1,j,i,j+1,out,t))
			t[i,j] = dst
			out[i,j] = 1
			heapOut.put((dst,(i,j)))
		if(out[r+1,c] == 2):
			i = r+1
			j = c
			dst = min4(fms(i-1,j,i,j-1,out,t),fms(i+1,j,i,j-1,out,t),fms(i-1,j,i,j+1,out,t),fms(i+1,j,i,j+1,out,t))
			t[i,j] = dst
			out[i,j] = 1
			heapOut.put((dst,(i,j)))
		if(out[r,c+1] == 2):
			i = r
			j = c+1
			dst = min4(fms(i-1,j,i,j-1,out,t),fms(i+1,j,i,j-1,out,t),fms(i-1,j,i,j+1,out,t),fms(i+1,j,i,j+1,out,t))
			t[i,j] = dst
			out[i,j] = 1
			heapOut.put((dst,(i,j)))

	for (r,c) , val in np.ndenumerate(out):
		if(val == 3):
			#out[r,c] = 0
			t[r,c] = -t[r,c]

	print 'Preprocessing Completed'
	telea(mask,t,ret,epsilon,heap)
	print 'Completed'

	return (ret[1:-1,1:-1,:]).astype('uint8')

def inpaint(img,mask,epsilon):
	#if(len(img.shape) == 2):
	# img = inPaint(img,mask,epsilon)
	# else:
	# 	#img = cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
	# 	img[:,:,0] = inPaint(img[:,:,0],mask,epsilon)
	# 	# print 'B done'
	# 	img[:,:,1] = inPaint(img[:,:,1],mask,epsilon)
	# 	# print 'G done'
	# 	img[:,:,2] = inPaint(img[:,:,2],mask,epsilon)
	# 	# print 'R done'
	#	# img = cv2.cvtColor(img,cv2.COLOR_LUV2BGR)
	# cv2.imwrite(output_string,img)
	return inPaint(img,mask,epsilon)
