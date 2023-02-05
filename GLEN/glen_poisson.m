function [L,Y,offset,L_iter,O_iter,Y_iter] = glen_poisson(X_noisy,param)

N = size(X_noisy,1);
M = size(X_noisy,2);
max_iter = param.max_iter;
alpha = M*param.alpha;
beta = param.beta;
gamma = param.gamma;
vi = param.vi;
reg_type = param.reg_type;

% define Y_0
Y_0 = X_noisy;

% initialize offset
offset = log(mean(Y_0,2));
offset = offset - 0.5*log((var(Y_0,0,2)-mean(Y_0,2))./mean(Y_0,2).^2+1);

% initialize Y
Y = log(Y_0+1)-mean(log(Y_0+1),2);
Y = Y - mean(Y,1);

V = ones(N,M)*param.vi;

for i = 1:max_iter

    Y_iter(:,:,i) = Y;
    
    % Step 1: given Y, update L
    if (isfield(param, 'L_init') && i==1)
        L = param.L_init;
    else
        switch reg_type
            case 'lasso'
                L = graph_learning_logdet_reglap(Y,param);
            case 'l2'
                Z = gsp_distanz(Y').^2;
                W = gsp_learn_graph_l2_degrees(Z/beta/4/M,1,struct("maxit",200));
                W(W<0) = 0;
                L = diag(sum(W,1))-W;
            case 'log'
                Z = gsp_distanz(Y').^2;
                W = gsp_learn_graph_log_degrees(Z/beta/2/M,1,gamma/beta/2);
                W(W<0) = 0;
                L = diag(sum(W,1))-W;
            case 'cgl'
                S = (Y*Y')/M;
                S = S + diag(sum(V,2))/M;
                A_mask = ones(size(S))-eye(size(S));
                L = estimate_cgl(S,A_mask,beta,1e-4,1e-6,40,1);
        end
    end
    if any(isnan(L),'all')
        if i == 1
            Y = zeros(N, M);
            O_iter(:,i) = zeros(N, 1);
            L = eye(N)-1/N;
            L_iter(:,:,i) = L;
        end
        return
    end
    L_iter(:,:,i) = L;

    % Step 3: Update offset
    for j = 1:5
        grad = sum(- Y_0 + exp(offset+Y+V/2), 2);
        hess = sum(exp(offset+Y+V/2), 2);
        offset = offset - grad./hess;
    end
    O_iter(:,i) = offset;


    % Step 2: Given L, update Y
    for j = 1:1
        grad = - Y_0 + exp(offset+Y+V/2) + alpha/M * L * Y;
        hess = alpha/M * L + (eye(N).*repmat(reshape(exp(offset+Y+V/2),[1,N,M]),[N,1,1]));
        grad_ct = [grad; zeros(1,M)];
        hess_ct = ones(N+1,N+1,M);
        hess_ct(1:N,1:N,:) = hess;
        hess_ct(end,end,:) = 0;
        D = -pagemldivide(hess_ct,reshape(grad_ct,[N+1,1,M]));
        t = 0.5;
        Y = Y+t*squeeze(D(1:N,:));
    end

%     % Step 3: Given L, update V
%     for j = 1:1
%         grad = (-1./V + exp(offset+Y+V/2) + diag(L)) / 2;
%         hess = eye(N).*repmat(reshape((1./V.^2 + exp(offset+Y+V/2)/2)/2, [1,N,T]),[N,1,1]);
%         D = -pagemldivide(hess,reshape(grad,[N,1,T]));
%         t = 0.5;
%         V = V+t*squeeze(D);
%         V(V<0) = 1e-4;
%     end

end
end