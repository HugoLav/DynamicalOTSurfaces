nVertices = Vertices.shape[0]

mub0 = np.zeros(nVertices)
mub1 = np.zeros(nVertices)

for i in range(nVertices):

    mub0[i] += areaVertices[i] * cut_off.f(-Vertices[i, 0] + 0.1, 0.15)
    mub1[i] += areaVertices[i] * cut_off.f(Vertices[i, 0] + 0.1, 0.15)

# Normalization
mub0 /= np.sum(mub0)
mub1 /= np.sum(mub1)
