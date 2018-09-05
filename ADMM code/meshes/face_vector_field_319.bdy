nVertices = Vertices.shape[0]

mub0 = np.zeros(nVertices)
mub1 = np.zeros(nVertices)

lengthScale = 0.1 

# Center of the "blobs" for mub0 and mub1
center0 = Vertices[4492,:]
center1 = Vertices[4225,:]

for i in range(nVertices) :
	
	# Change of coordinates 
	alpha = 0.1 * Vertices[i,0] + Vertices[i,1] 
	beta = - Vertices[i,0] + 0.1 * Vertices[i,1] 
	gamma = Vertices[i,2]

	# Define mub0
	if gamma >= -0.1 :		
		mub0[i] = areaVertices[i] * cut_off.f(-0.2-alpha,0.3) * cut_off.f(alpha - 0.15,0.3) * cut_off.f(0.1 - beta,0.3) * cut_off.f(beta - 0.45,0.3)
		
	# Define mub1 
	mub1[i] += areaVertices[i] * exp( -lin.norm( Vertices[i,:] - center0  )**2 / lengthScale**2  )
	mub1[i] += areaVertices[i] * exp( -lin.norm( Vertices[i,:] - center1  )**2 / lengthScale**2  )
		
# Normalization 
mub0 /= np.sum(mub0)
mub1 /= np.sum(mub1)
