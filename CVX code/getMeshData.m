function mesh = getMeshData(X,T,numEigs,name)
% Creates a mesh object storing various useful geometric quantities
% 
% Input:
%	X - ordered list of vertices and their positions
%	T - ordered list of triangles by vertices (numerical indices to X)
%	numEigs - number of Laplace-Beltrami eigenfunctions desired
%	name - a desired mesh name

if nargin < 4
    name = 'mesh';
end

if nargin < 3
    numEigs = 0;
end

mesh = [];
mesh.vertices = X;
mesh.triangles = double(T);
mesh.name = name;

[mesh.cotLaplacian,mesh.areaWeights] = cotLaplacian(X,T);

% Change to negative cot Laplacian and rescale to area = 1
mesh.areaWeights = mesh.areaWeights / sum(mesh.areaWeights);
mesh.cotLaplacian = -1*mesh.cotLaplacian;

mesh.numVertices = size(X,1);
mesh.numTriangles = size(T,1);

evec = [mesh.triangles(:,1) mesh.triangles(:,2)];
evec = [evec; mesh.triangles(:,2) mesh.triangles(:,1)];
evec = [evec; mesh.triangles(:,1) mesh.triangles(:,3)];
evec = [evec; mesh.triangles(:,3) mesh.triangles(:,1)];
evec = [evec; mesh.triangles(:,2) mesh.triangles(:,3)];
evec = [evec; mesh.triangles(:,3) mesh.triangles(:,2)];
evec = unique(evec,'rows');

mesh.edges = evec(evec(:,1) < evec(:,2),:);
mesh.numEdges = size(mesh.edges,1);

% Compute LB eigenstuff
areaMatrix = sparse(1:mesh.numVertices,1:mesh.numVertices,mesh.areaWeights);

if numEigs > 0
    [evecs, evals] = eigs(mesh.cotLaplacian, areaMatrix, max(numEigs,1), -1e-5);
    evals = diag(evals);
    mesh.laplaceBasis = evecs;
    mesh.eigenvalues = evals;
end

normalf = cross( mesh.vertices(mesh.triangles(:,2),:)'-mesh.vertices(mesh.triangles(:,1),:)', ...
                 mesh.vertices(mesh.triangles(:,3),:)'-mesh.vertices(mesh.triangles(:,1),:)' );
d = sqrt( sum(normalf.^2,1) ); d(d<eps)=1;
mesh.faceNormals = (normalf ./ repmat( d, 3,1 ))';
