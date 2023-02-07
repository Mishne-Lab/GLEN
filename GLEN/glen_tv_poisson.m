function [L,Y,offset,convergence,L_iter, O_iter] = glen_tv_poisson(X_noisy,param)

N = size(X_noisy,1);
T = size(X_noisy,2);
max_iter = param.max_iter;
alpha = T*param.alpha;
beta = param.beta;
gamma = param.gamma;
reg_type = param.reg_type;

objective = zeros(max_iter,1);
convergence = 1;

% define Y_0
Y_0 = X_noisy;

% initialize offset
offset = mean(log(Y_0+1),2);
offset = offset - 0.5*log((var(Y_0,0,2)-mean(Y_0,2))./mean(Y_0,2).^2+1);

% initialize Y
Y = log(Y_0+1)-mean(log(Y_0+1),2);
Y = Y - mean(Y,1);

if isfield(param, 'Y_init')
    Y = param.Y_init;
end

for i = 1:max_iter
    
    % Step 1: given Y, update L
    if (isfield(param, 'L_init') && i==1)
        L = param.L_init;
    else
        switch reg_type
            case 'lasso'
                L = graph_learning_logdet_reglap(Y,param);
            case 'l2'
                Z = gsp_distanz(Y').^2;
                W = gsp_learn_graph_l2_degrees(Z/beta/4/T,1,struct("maxit",200));
                W(W<0) = 0;
                L = diag(sum(W,1))-W;
            case 'log'
                Z = gsp_distanz(Y').^2;
                W = gsp_learn_graph_log_degrees(Z/beta/2/T,1,gamma/beta/2);
                W(W<0) = 0;
                L = diag(sum(W,1))-W;
            case 'cgl'
                S = (Y*Y')/T;% + diag(sum(exp(V),2))/T;
                A_mask = ones(size(S))-eye(size(S));
                L = estimate_cgl(S,A_mask,beta,1e-4,1e-6,40,1);
        end
    end
    if any(isnan(L),'all')
        convergence = 0;
        if i == 1
            Y = zeros(N, T);
            O_iter(:,i) = zeros(N, 1);
            L = eye(N)-1/N;
            L_iter(:,:,i) = L;
        end
        L = L_iter(:,:,end);
        return
    end
    L_iter(:,:,i) = L;

    % Step 3: Update offset
    for j = 1:5
        grad = sum(- Y_0 + exp(offset+Y), 2);
        hess = sum(exp(offset+Y), 2);
        offset = offset - grad./hess;
    end
    O_iter(:,i) = offset;

    % Step 2: Given L, update Y
    for j = 1:1
        grad = - Y_0 + exp(offset+Y) + alpha/T * L * Y + gamma*(2*Y-[Y(:,2:end),Y(:,end)]-[Y(:,1),Y(:,1:end-1)]);
        hess = alpha/T * L + (eye(N).*repmat(reshape(exp(offset+Y),[1,N,T]),[N,1,1])) + 2*gamma*eye(N);
        grad_ct = [grad; zeros(1,T)];
        hess_ct = ones(N+1,N+1,T);
        hess_ct(1:N,1:N,:) = hess;
        hess_ct(end,end,:) = 0;
        D = -pagemldivide(hess_ct,reshape(grad_ct,[N+1,1,T]));
        t = 0.5;
        Y = Y+t*squeeze(D(1:N,:));
    end

end
end