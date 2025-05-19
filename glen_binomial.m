function [L,Y,llp] = glen_binomial(X_noisy,param)

N = size(X_noisy,1);
M = size(X_noisy,2);
max_iter = param.max_iter;
max_iter_inner = param.max_iter_inner;
step_size = param.step_size;
tol = param.tol;
alpha = param.alpha;
beta = param.beta;
gamma = param.gamma;
lsolver = param.lsolver;
K = param.K;

if isfield(param, 'h_type')
    h_type = param.h_type;
else
    h_type = "type2";
end
switch h_type
    case "type1"
        H = 2*eye(N)-ones(N,N);
    case "type2"
        H = eye(N)-ones(N,N);
end
J = ones(N,N)/N;

% define Y_0
Y_0 = X_noisy;
Y = (1+X_noisy)/(K+2);
Y = log(Y./(1-Y));
Y = Y-mean(Y,1);

switch param.init
    case 'cgl'
        S = (Y*Y')/M;
        A_mask = ones(size(S))-eye(size(S));
        L = estimate_cgl(S,A_mask,alpha,1e-4,1e-6,40,1);
end
A = -L+diag(diag(L));
w = A(tril(true(N),-1));

llp = zeros(max_iter,1);
for i = 1:max_iter
    
    % Step 1: given Y, update L
    if (isfield(param, 'L_init') && i==1)
        L = param.L_init;
    else
        switch lsolver
            case 'lasso'
                L = graph_learning_logdet_reglap(Y,param);
            case 'l2'
                Z = gsp_distanz(Y').^2;
                W = gsp_learn_graph_l2_degrees(Z/beta/4/M,1,struct("maxit",200));
                W(W<0) = 0;
                L = diag(sum(W,1))-W;
            case 'log'
                Z = 5*gsp_distanz(Y').^2/M;
                W = gsp_learn_graph_log_degrees(Z,gamma,beta);
                W(W<0) = 0;
                L = diag(sum(W,1))-W;
            case 'cgl'
                S = (Y*Y')/M;
                A_mask = ones(size(S))-eye(size(S));
                L = estimate_cgl(S,A_mask,beta,1e-4,1e-6,40,1);
            case 'gd'
                S = (Y*Y')/M;
                for k = 1:max_iter_inner
                    w0 = w;
                    L_gd = -gamma*inv(J+L)+S+beta*H;
                    w_gd = Lstar(L_gd);
                    w = w-step_size*w_gd;
                    w(w<0) = 0;
                    if norm(w-w0,2)<tol
                        break
                    end
                    A = squareform(w);
                    L = diag(sum(A,1))-A;
                end
        end
    end
    A = -L+diag(diag(L));

    % Step 2: Given L, update Y
    for j = 1:max_iter_inner
        grad = - Y_0 + K./(1+exp(-Y)) + L * Y;
        hess = L + (eye(N).*repmat(reshape(K./(exp(Y)+2+exp(-Y)),[1,N,M]),[N,1,1]));
        grad_ct = [grad; zeros(1,M)];
        hess_ct = ones(N+1,N+1,M);
        hess_ct(1:N,1:N,:) = hess;
        hess_ct(end,end,:) = 0;
        D = -pagemldivide(hess_ct,reshape(grad_ct,[N+1,1,M]));
        t = 0.5;
        Y = Y+t*squeeze(D(1:N,:));
    end
    
    llp(i) = sum(Y_0.*Y/M + K/M*log(1./(1+exp(Y))),'all') - 0.5*trace(Y'*L*Y)/M + 0.5*logdet(L+J);
    % yterm(i) = sum(Y_0.*Y/M + K/M*log(1./(1+exp(Y))),'all') - 0.5*trace(Y'*L*Y)/M;

end

end