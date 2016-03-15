import networkx as nx
import numpy as np
import math

d = 0.85
alpha = 0.7
step_size = 0.1
iteration = 0

B = np.array([[0, 1, 0, 0, -1], [0, -1, 1, 0, 0]])
print B
miu = np.array([0.8, 0.2])

# features of edges
x = {
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
y = {
1 : np.array([2, 4, 3]),
2 : np.array([2, 4, 5]),
3 : np.array([3, 3, 4]),
4 : np.array([3, 4, 6]),
5 : np.array([3, 4, 2]),
}
phi = np.array([0.4, 0.3, 0.3])

n = len(y) # #nodes
h = len(phi) # #node features
pi = np.full(n, 1.0/n)
print "pi first:", pi
e = 1

while e > 0.01:

###########################################################
	# in loop
	DG = nx.DiGraph()
	for key in x:
		# xweight[key] = np.dot(x[key], omega)
		DG.add_weighted_edges_from([(key / 10, key % 10, 
			np.dot(x[key], omega.T))])

	# Note: P is a row major matrix
	P = nx.to_numpy_matrix(DG)
	# not good enough, have to handle dangling nodes
	P /= P.sum(axis=1)
	P = np.array(P)
	print "P:", P
############################################################
#############################################################
	yweight = {} # a dict
	r_temp = []
	for key in y:
		yweight[key] = np.dot(y[key], phi.T)	
		r_temp.append(yweight[key])
	r = np.array(r_temp)
	r = r.T
	r /= r.sum()
	print "r:",r
	pi3 = d * np.dot(P.T, pi.T) + (1 - d) * r - pi
	print "pi3:", pi3
#################################################################
	P1 = d * P - np.identity(n)
	print "P1:", P1
	# print "***********TEST************"
	# print np.dot(P1, pi3.T)
	# print "************TESTEND********"
	g_pi = (1 - alpha) * np.dot(P1, pi3.T).T - alpha/2 * np.dot(B.T, miu.T)
	print "g_pi:", g_pi
#######################################################################
#########################################################################
	sumjk = []
	for i in range(1, n+1):
		sumjk_temp = 0
		for key in x:
			if key/10 == i:
				sumjk_temp += np.dot(x[key], omega.T)
		sumjk.append(sumjk_temp)
	print "sumjk:", sumjk

	sumk = {}
	for key in x:
		sumk[key] = np.dot(x[key], omega.T)
	print "sumk:", sumk

	sumj = [([0] * l) for i in range(n)]
	sumj_temp = 0
	for k in range(0, l):
		for i in range(1, n+1):
			sumj_temp = 0
			for key in x:
				if key/10 == i:
					sumj_temp += x[key][k]
					sumj[i-1][k] = sumj_temp
	print "sumj:", sumj

	def get_W(i, j, k):
		ij = 10 * (i + 1) + j + 1
		# print "ij:", ij
		# print "xijk:", x[12][0]
		if ij in x:
			result = (x[ij][k] * sumjk[i] - sumk[ij] * sumj[i][k]) / (sumjk[i] * sumjk[i])
		else:
			result = 0
		return result

	W_temp = [([0] * l) for i in range(n * n)]
	for i in range(n*n):
		for j in range(l):
			W_temp[i][j] = get_W(i%10, i/10, j)
	print "W_temp:", W_temp

	W = np.asarray(W_temp) # W is partial derivatives
	g_omega = (1 - alpha) * np.dot((np.kron(pi3, pi)), W)
	print "g_omega:", g_omega
######################################################################
########################################################################3
	R_temp = []
	for key in y:
		R_temp.append(y[key])
	R = np.array(R_temp)
	print "R:", R

	g_phi = (1 - alpha) * (1 - d) * np.dot(pi3, R)
	print "g_phi:", g_phi
##########################################################################
#########################################################################
	b = B.shape[0] # supervised information

	s = np.dot(miu, (np.ones(b) - np.dot(B, pi.T)).T)

	G1 = alpha * np.dot(pi3, pi3.T) + (1 - alpha) * s
##############################################################################
	print "pi before:", pi
	pi = pi - step_size * g_pi
	pi /= pi.sum()
	print "pi after:", pi
	print "omega before:", omega
	omega = omega - step_size * g_omega
	omega /= omega.sum()
	print "omega after:", omega
	print "phi before:", phi
	phi = phi - step_size * g_phi
	phi /= phi.sum()
	print "phi  after:", phi
	print "pi3square:", np.dot(pi3, pi3.T)

	s = np.dot(miu, (np.ones(b) - np.dot(B, pi.T).T).T)

	G2 = alpha * np.dot(pi3, pi3.T) + (1 - alpha) * s

	e = abs(G2 - G1)
	print "***************************************************************************************************************************"
	print "G1", G1
	print "G2", G2
	print "e:", e
	print "***************************************************************************************************************************"
	iteration += 1
	print "iteration:", iteration

print "RESULT:"
print "pi:", pi
print "omega:", omega
print "phi:", phi
print "total iteration:", iteration