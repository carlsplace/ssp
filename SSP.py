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

def calcPi3(node, phi, pi):
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

def calcGradientPi(pi, r, P, B, miu):
	P1 = d * P - np.identity(nv)
	g_pi = (1 - alpha) * np.dot(P1, pi3.T).T - alpha/2 * np.dot(B.T, miu.T)
	return g_pi

def get_W(i, j, k):
	ij = 10 * (i + 1) + j + 1
	if ij in x:
		result = (x[ij][k] * sumjk[i] - sumk[ij] * sumj[i][k]) / (sumjk[i] * sumjk[i])
	else:
		result = 0
	return result

def calcGradientOmega(x, omega):
	"complicated, holy shit..."
	sumjk = []
	for i in range(1, nv+1):
		sumjk_temp = 0
		for key in x:
			if key/10 == i:
				sumjk_temp += np.dot(x[key], omega.T)
		sumjk.append(sumjk_temp)

	sumk = {}
	for key in x:
		sumk[key] = np.dot(x[key], omega.T)

	sumj = [([0] * nvf) for i in range(n)]
	sumj_temp = 0
	for k in range(0, nvf):
		for i in range(1, n+1):
			sumj_temp = 0
			for key in x:
				if key/10 == i:
					sumj_temp += x[key][k]
					sumj[i-1][k] = sumj_temp

	W_temp = [([0] * nvf) for i in range(nv * nv)]
	for i in range(nv * nv):
		for j in range(nvf):
			W_temp[i][j] = get_W(i%10, i/10, j)

	W = np.array(W_temp) # W is partial derivatives
	g_omega = (1 - alpha) * np.dot((np.kron(pi3, pi)), W)
	return g_omega

def calcGradientPhi(pi3, y):
	R_temp = []
	for key in y:
		R_temp.append(y[key])
	R = np.array(R_temp)
	g_phi = (1 - alpha) * (1 - d) * np.dot(pi3, R)
	return g_phi

def calcG(pi3, B, miu):
	b = B.shape[0] # supervised information
	s = np.dot(miu, (np.ones(b) - np.dot(B, pi.T)).T)
	G = alpha * np.dot(pi3, pi3.T) + (1 - alpha) * s
	return G

# global variable: alpha, d, nv(num_of_nodes), nvf(num_of_nodefeatures)