#!/usr/bin/env
#author@Pranjal

import sys
import cv2

from inpaint import inpaint

input_list = sys.argv

# use eps around 5
eps = 5

if(len(input_list) == 4):
	l = cv2.imread(input_list[1],1)
	m = cv2.imread(input_list[2],0)
	img = inpaint(l,m,eps)
	cv2.imwrite(input_list[3],img)