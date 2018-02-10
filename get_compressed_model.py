import numpy as np
import caffe
import os
import sys

#net = caffe.Net('./examples/cifar10/cifar10_full_train_test.prototxt', weights='./examples/cifar10/cifar10_2_full_iter_60000.caffemodel.h5')


solver = caffe.AdamSolver('./examples/cifar10/cifar10_full_solver.prototxt')
solver_com = caffe.AdamSolver('./examples/cifar10/cifar10_comp_solver.prototxt')


solver.net.copy_from('./examples/cifar10/cifar10_full_iter_60000.caffemodel.h5')

prev = 0
for i in solver.net.params:
	if(i[0:4] == "conv"):
		#print(i)
		solver_com.net.params[i][0].data[...] = solver.net.params[i][0].data[solver.net.params[i][0].shape[0]//32*20:,prev:,:,:]
		solver_com.net.params[i][1].data[...] = solver.net.params[i][1].data[solver.net.params[i][1].shape[0]//32*20:]
		#for j in solver.net.params[i]:
		#	print(j.data)

		prev = solver.net.params[i][0].shape[0]//32*20
	if(i[0:2] == "ip"):
		solver_com.net.params[i][0].data[...] = solver.net.params[i][0].data[:,solver.net.params[i][0].shape[1]//32*20:]
		solver_com.net.params[i][1].data[...] = solver.net.params[i][1].data[...]


# for i in solver_com.net.params:
# 	print(i)
# 	for j in solver_com.net.params[i]:
# 		print(j.data)

solver_com.net.save_hdf5("compressed.h5")
