nVertices = Vertices.shape[0]

mub0 = np.zeros(nVertices)
mub1 = np.zeros(nVertices)

# Center of the "blob" for mub0 
x0 = np.array([0.33,0.5,0.0])
# Center of the "blob" for mub1 
x10 = np.array([0.8,0.2,0.0])
x11 = np.array([0.8,0.8,0.0])

for i in range(nVertices) :

	mub0[i] = areaVertices[i] * cut_off.f(lin.norm(Vertices[i,:] - x0) - 0.1 ,0.1)
	mub1[i] += areaVertices[i] * cut_off.f((lin.norm(Vertices[i,:] - x10) - 0.1)*2. ,0.1)
	mub1[i] += areaVertices[i] * cut_off.f((lin.norm(Vertices[i,:] - x11) - 0.1)*2. ,0.1)
	
# Normalization 
mub0 /= np.sum(mub0)
mub1 /= np.sum(mub1)
