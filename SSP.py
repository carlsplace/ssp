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

def calcPi3(node, phi, pi, P):
	# r is the reset probability vector, pi3 is an important vertor for later use
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

def calcGradientPi(pi, P, B, miu):
	P1 = d * P - np.identity(nv)
	g_pi = (1 - alpha) * np.dot(P1, pi3.T).T - alpha/2 * np.dot(B.T, miu.T)
	return g_pi

def get_W(i, j, k, sumj, sumk, sumjk):
	ij = 10 * (i + 1) + j + 1
	if ij in edge:
		result = (edge[ij][k] * sumjk[i] - sumk[ij] * sumj[i][k]) / (sumjk[i] * sumjk[i])
	else:
		result = 0
	return result

def calcGradientOmega(edge, omega):
	# complicated, holy shit...
	sumjk = []
	for i in range(1, nv+1):
		sumjk_temp = 0
		for key in edge:
			if key/10 == i:
				sumjk_temp += np.dot(edge[key], omega.T)
		sumjk.append(sumjk_temp)

	sumk = {}
	for key in edge:
		sumk[key] = np.dot(edge[key], omega.T)

	sumj = [([0] * nef) for i in range(nv)]
	sumj_temp = 0
	for k in range(0, nef):
		for i in range(1, nv+1):
			sumj_temp = 0
			for key in edge:
				if key/10 == i:
					sumj_temp += edge[key][k]
					sumj[i-1][k] = sumj_temp

	W_temp = [([0] * nef) for i in range(nv * nv)]
	for i in range(nv * nv):
		for j in range(nef):
			W_temp[i][j] = get_W(i%10, i/10, j, sumj, sumk, sumjk)

	W = np.array(W_temp) # W is partial derivatives
	g_omega = (1 - alpha) * np.dot((np.kron(pi3, pi)), W)
	return g_omega

def calcGradientPhi(pi3, node):
	R_temp = []
	for key in node:
		R_temp.append(node[key])
	R = np.array(R_temp)
	g_phi = (1 - alpha) * (1 - d) * np.dot(pi3, R)
	return g_phi

def calcG(pi3, B, miu):
	b = B.shape[0] # supervised information
	s = np.dot(miu, (np.ones(b) - np.dot(B, pi.T)).T)
	G = alpha * np.dot(pi3, pi3.T) + (1 - alpha) * s
	return G

def updateVar(var, g_var, step_size):
	var = var - step_size * g_var
	var /= var.sum()
	return var

# global variable: alpha, d, nv(num_of_nodes), nvf(num_of_nodefeatures)
d = 0.85
alpha = 0.5
step_size = 0.1
iteration = 0

B = np.array([[0, 1, 0, 0, -1], [0, -1, 1, 0, 0]])
miu = np.array([0.8, 0.2])

# features of edges
edge = {
12 : np.array([5, 7]),
13 : np.array([4, 5]),
23 : np.array([6, 4]),
42 : np.array([7, 4]),
34 : np.array([3, 7]),
41 : np.array([7, 6]),
35 : np.array([4, 6]),
45 : np.array([5, 4]),
52 : np.array([6, 6]),
}
omega = np.array([0.6, 0.4])
l = len(omega) 
# features of nodes
node = {
1 : np.array([2, 4, 3]),
2 : np.array([2, 4, 5]),
3 : np.array([3, 3, 4]),
4 : np.array([3, 4, 6]),
5 : np.array([3, 4, 2]),
}
phi = np.array([0.4, 0.3, 0.3])

nv = len(node) # #nodes
nvf = len(phi) # #node features
nef = len(omega) # #edge features
pi = np.full(nv, 1.0/nv)

e = 1
DG = createDirectedGraph(edge, omega)
P = getTransMatrix(DG)
pi3 = calcPi3(node, phi, pi, P)
G0 = calcG(pi3, B, miu)
while e > 0.00001:
	g_pi = calcGradientPi(pi, P, B, miu)
	g_omega =calcGradientOmega(edge, omega)
	g_phi = calcGradientPhi(pi3, node)
	pi = updateVar(pi, g_pi, step_size)
	omega = updateVar(omega, g_omega, step_size)
	phi = updateVar(phi, g_phi, step_size)
	DG = createDirectedGraph(edge, omega)
	P = getTransMatrix(DG)
	pi3 = calcPi3(node, phi, pi, P)
	G1 = calcG(pi3, B, miu)
	e = abs(G1 - G0)
	G0 = G1
	iteration += 1

print "RESULT:"
print "pi:", pi
print "omega:", omega
print "phi:", phi
print "total iteration:", iteration