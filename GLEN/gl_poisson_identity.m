function [L,Y,L_harvard,offset,L_iter] = gl_poisson_identity(X_noisy,param)
% Learning graphs (Laplacian) from structured signals
% Signals X follow Gaussian assumption

N = size(X_noisy,1);
T = size(X_noisy,2);
max_iter = param.max_iter;
alpha = T*param.alpha;
beta = param.beta;
gamma = param.gamma;
rho = param.rho;
damped = 0;
reg_type = param.reg_type;

objective = zeros(max_iter,1);

% define Y_0
Y_0 = X_noisy;

% fit offset
% offset = log(mu);
% offset = zeros(N,1);
offset = 5*ones(N,1);

% define initial Y
Y = Y_0 - mean(Y_0,1);

L_iter = zeros(N,N,5);

for i = 1:max_iter
    if any(isnan(Y))
        return
    end
    
    % Step 1: given Y, update L
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
            S = (Y*Y')/T;
            A_mask = ones(size(S))-eye(size(S));
            L = estimate_cgl(S,A_mask,beta,1e-4,1e-6,40,1);
    end
    L_iter(:,:,i) = L;
%     sum(exp(offset+Y+exp(V)/2),'all') - sum(Y_0.*(offset+Y),'all') - sum(V, 'all')/2 + alpha/2*vec(Y*Y')'*vec(L)/T + alpha*beta*trace(L) - alpha/2*log(det(L+1/N))

    % solution in the Harvard paper
    if i == 1
        L_harvard = L;
    end
    
    if any(isnan(L),'all')
        Y = zeros(N, T);
        offset = zeros(N, 1);
        return
    end

    % Step 2: Given L, update Y
    grad = - Y_0./(Y+offset) + 1 + alpha/T * L * Y;
    hess = alpha/T * L + (eye(N).*repmat(reshape(Y_0./(Y+offset).^2,[1,N,T]),[N,1,1]));
    grad_ct = [grad; zeros(1,T)];
    hess_ct = ones(N+1,N+1,T);
    hess_ct(1:N,1:N,:) = hess;
    hess_ct(end,end,:) = 0;
    D = -pagemldivide(hess_ct,reshape(grad_ct,[N+1,1,T]));
    t = 0.5;
    Y = Y+t*squeeze(D(1:N,:));

    % plot the objective
    switch reg_type
        case 'l2'
            objective(i) = sum(Y,'all') - sum(Y_0.*log(offset+Y),'all') + alpha/2*vec(Y*Y')'*vec(L)/T + alpha/2*beta*(norm(L,'fro')^2);% + rho/2 * norm(Y,'fro')^2;
        case 'log'
            objective(i) = sum(Y,'all') - sum(Y_0.*log(offset+Y),'all') + alpha/2*vec(Y*Y')'*vec(L)/T - alpha/2*gamma*sum(log(diag(L))) + alpha/2*beta*(norm(L-diag(diag(L)),'fro')^2);
        case 'cgl'
            objective(i) = sum(Y,'all') - sum(Y_0.*log(offset+Y),'all') + alpha/2*vec(Y*Y')'*vec(L)/T + alpha*beta*trace(L) - alpha/2*log(det(L+1/N));% + rho/2 * norm(Y,'fro')^2;
    end
%     sum(exp(offset+Y+exp(V)/2),'all') - sum(Y_0.*(offset+Y),'all') - sum(V, 'all')/2 + alpha/2*vec(Y*Y')'*vec(L)/T + alpha*beta*trace(L) - alpha/2*log(det(L+1/N))
    figure(4)
    plot(i,objective(i), '.r');
    hold on, drawnow
    
    % stopping criteria
    if i == 10
        return
    end
    if i>=2 && abs(objective(i)-objective(i-1))/abs(objective(i-1))<10^(-6)
        return
    end

    L_prev = L;

end
end