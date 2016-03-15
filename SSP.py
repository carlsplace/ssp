import networkx as nx
import numpy as np
import math

def createDirectedGraph(edge, omega):
	DG = nx.DiGraph()
	for key in edge:
		DG.add_weighted_edges_from([(key / 10, key % 10,
			np.dot(edge[key], omega.T))])
	return DG

def getTransMatrix(DG):
	P = nx.to_numpy_matrix(DG)
	# not good enough, have to handle dangling nodes
	P /= P.sum(axis=1)
	P = np.array(P)
	return P

def calcPi3(node, phi, pi, d=0.85):
	"r is the reset probability vector, pi3 is an important vertor for later use"
	node_weight = {}
	r_temp =[]
	for key in node:
		node_weight[key] = np.dot(node[key], phi.T)
		r_temp.append(node_weight[key])
	r = np.array(r_temp)
	r = r.T
	r /= r.sum()
	pi3 = d * np.dot(P.T, pi.T) + (1 - d) * r - pi
	return pi3

def calcGradientPi(pi, r):
	pass

def calcGradientOmega():
	"complicated"
	pass

def calcGradientPhi():
	pass

def calcG():
	pass