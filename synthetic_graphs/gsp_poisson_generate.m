nreplicate = 20; % repeat the same experiment (based on different graphs)
for ii = 1:nreplicate

%% generate graph topology
param.N = 20;
[A,XCoords, YCoords] = construct_graph(param.N,'er',0.3);
% [A,XCoords, YCoords] = construct_graph(param.N,'sbm',0.4,0.1);
% [A,XCoords, YCoords] = construct_graph(param.N,'ws',2,0.1);

%% generate weighted graphs
L_0 = full(sgwt_laplacian(A,'opt','raw'));
A_0 = -L_0+diag(diag(L_0));
W = triu(rand(param.N)*1.9 + 0.1, 1);
A_0 = A_0 .* (W+W');
L_0 = diag(sum(A_0)) - A_0;
L_0 = L_0/trace(L_0)*param.N;

%% generate smooth signals on graphs
[V,D] = eig(full(L_0));
sigma = pinv(D);
mu = zeros(1,param.N);
num_of_signal = 2000;
param.M = num_of_signal;
gftcoeff = mvnrnd(mu,sigma,num_of_signal);
X = V*gftcoeff';

%% generate noisy signals on graphs
offset = [2*ones(param.N/2,1);-2*ones(param.N/2,1)];
X_noisy = poissrnd(exp(X+offset), size(X));

%% save
data{ii,1} = A;
data{ii,2} = L_0;
data{ii,3} = X;
data{ii,4} = X_noisy;

end


%% construct the graph
function [G, XCoords, YCoords] = construct_graph(N,opt,varargin1,varargin2)

switch opt
    case 'gaussian', % random geometric graph with Gaussian weights
        plane_dim = 1;
        XCoords = plane_dim*rand(N,1);
        YCoords = plane_dim*rand(N,1);

        T = varargin1; 
        s = varargin2;
        d = distanz([XCoords,YCoords]'); 
        W = exp(-d.^2/(2*s^2)); 
        W(W<T) = 0; % Thresholding to have sparse matrix
        W = 0.5*(W+W');
        G = W-diag(diag(W));
        
    case 'er', % Erdos-Renyi random graph
        p = varargin1;
        G = erdos_reyni(N,p);
        
    case 'pa', % scale-free graph with preferential attachment
        m = varargin1;
        G = preferential_attachment_graph(N,m);
        
    case 'ff', % forest-fire model
        p = varargin1;
        r = varargin2;
        G = forest_fire_graph(N,p,r);
        
    case 'chain' % chain graph
        G = spdiags(ones(N-1,1),-1,N,N);
        G = G + G';

    case 'sbm' % stochastic block model
        p = varargin1;
        q = varargin2;
        G1 = erdos_reyni(N/2,p);
        G2 = erdos_reyni(N/2,p);
        G3 = rand(N/2)<q;
        G = [G1,G3;G3',G2];

    case 'ws' % small world network
        % Connect each node to its K next and previous neighbors. This constructs
        % indices for a ring lattice.
        K = varargin1;
        p = varargin2;
        s = repelem((1:N)',1,K);
        t = s + repmat(1:K,N,1);
        t = mod(t-1,N)+1;

        % Rewire the target node of each edge with probability p
        for source=1:N    
            switchEdge = rand(K, 1) < p;
            
            newTargets = rand(N, 1);
            newTargets(source) = 0;
            newTargets(s(t==source)) = 0;
            newTargets(t(source, ~switchEdge)) = 0;
            
            [~, ind] = sort(newTargets, 'descend');
            t(source, switchEdge) = ind(1:nnz(switchEdge));
        end

        G = zeros(N);
        idx = sub2ind(size(G), s(:),t(:));
        G(idx) = 1;
        G = G + G';
end
end