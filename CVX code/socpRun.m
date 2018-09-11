function [mu,runtime] = socpRun(dataFile,N,alpha)
% An approximate geodesic in the Wasserstein space is calculated, with a 
% mesh and endpoint distributions from "dataFile". N timesteps are
% calculated, with alpha as a congestion regularization parameter.
%
% More specifically, the dual problem from eqns. (25)-(26) is solved with 
% CVX. Mosek was used as the solver for the experiments and figures in the 
% paper. Default precision settings were used.
%
% Inputs: 
%   dataFile - char string for a .mat object containing mesh and
% start/target distributions
%   N - number of time steps to calculate
%   alpha - regularization parameter, greater means more regularization
%
% Outputs:
%   mu - (nv x N) matrix containing the interpolated distributions
%   runtime - total runtime of the algorithm

% M is a mesh data object, created with getMeshData.m
% mu0 is the start distribution
% mu1 is the end distribution
load(dataFile,'M','mu0','mu1');

% In the dual objective, there is a 1/alpha term, which is given by
% alphaCVX. To avoid division by zero, we set alphaCVX to be large (10e6) 
% when alpha is very small.
if alpha >= 0.000001
    alphaCVX = 1/alpha;
else
    alphaCVX = 1000000;
end

tic % start timer

FEM = firstOrderFEM(M); % creates FEM diff. operators

nv = M.numVertices;
nt = M.numTriangles;
tau = 1/N; % time step size

% Time derivative operator
D = sparse([1:N 1:N], [1:N 2:(N+1)], [-ones(1,N) ones(1,N)], N, N+1)/tau;

% Spatial averaging operator
row = M.triangles;
col = [(1:nt)' (1:nt)' (1:nt)'];
val = [FEM.faceAreas FEM.faceAreas FEM.faceAreas];
spatialAverage = sparse(double(row),double(col),val,nv,nt);
rowSums = sum(spatialAverage,2);
spatialAverage = spatialAverage ./ rowSums;

areaWtsN = repmat(M.areaWeights,N,1); % area weights replicated N times

% Call CVX
cvx_begin
    cvx_solver mosek
    
    variable phi(nv,N+1)
    variable normMtx(nt,N+1) % dummy variable for norm squared of gradients of phi 
    variable lambda(nv,N) % dual variable associated with congestion constraint
    variable regterm(1,1) % regularization term
    dual variable mu


    % The formulation below contains a few relaxations that will be strict 
    % upon optimization, as well as some reshuffling of constraints to
    % improve CVX's initial processing spee.
    % Recall again that alphaCVX = 1/alpha from the text.
    maximize sum(M.areaWeights.*mu1.*phi(:,end)) - sum(M.areaWeights.*mu0.*phi(:,1)) - .5 * alphaCVX * regterm
    subject to
        % calculating time derivative
        timeDerivative = phi * D';
        
        % calculating gradient components
        dx1 = FEM.grad{1} * phi;
        dx2 = FEM.grad{2} * phi;
        dx3 = FEM.grad{3} * phi;
        
        % SOCP version of:  normMtx >= dx1.^2+dx2.^2+dx3.^2;
        norms([dx1(:) dx2(:) dx3(:) (1-normMtx(:))/2],2,2) <= (1+normMtx(:))/2
        
        % averaging gradient norms
        timeAverage = (normMtx(:,1:N)+normMtx(:,2:(N+1)))/2;
        spacetimeAverage = spatialAverage*timeAverage;
        
        % main constraint
        mu : timeDerivative + .5 * spacetimeAverage <= lambda;
        
        % for regularization term
        % SOCP version of: sum((lambda(:).^2).*areaWtsN)./N <= regterm;
        norm([(1-regterm)/2 ; lambda(:).*sqrt(areaWtsN/N)]) <= (1+regterm)/2
cvx_end

% As CVX does not take the appropriate inner product when forming the
% Lagrangian, one must divide and multiply appropriately to obtain the 
% interpolating 2-form.
mu = (mu*N)./M.areaWeights;

fprintf('Objective: %.12f\n', full(sum(M.areaWeights.*mu1.*phi(:,end)) - sum(M.areaWeights.*mu0.*phi(:,1))));
fprintf('Regularization: %.12f\n', .5 * alphaCVX * regterm);

runtime = toc; % end timer and record

end
