function [L,Y,offset,L_iter,O_iter] = gl_bernoulli_log(X_noisy,param)
% Learning graphs (Laplacian) from structured signals
% Signals X follow Gaussian assumption

N = size(X_noisy,1);
T = size(X_noisy,2);
% m = param.m; % num of mc samples
max_iter = param.max_iter;
alpha = T*param.alpha;
beta = param.beta;
gamma = param.gamma;
reg_type = param.reg_type;

% objective = zeros(max_iter,1);

% define Y_0
Y_0 = X_noisy;

% fit offset
% offset = zeros(N,1);

% oracle
% offset = [-1*ones(N/2,1);-2*ones(N/2,1)];
offset = log(mean(Y_0,2)./(1-mean(Y_0,2)));

% define initial Y
% Y = log(Y_0+1)-mean(log(Y_0+1),1);
% Y = log(Y_0+1)-mean(log(Y_0+1),2);
% Y = Y - mean(Y,1);

Y = Y_0 - mean(Y_0,2);
Y = Y - mean(Y,1);
% Y = Y - offset;
% Y = log(Y_0+1);
% Y = Y - mean(Y,1);
% Y = log(exp(log(Y_0+1)+0.5*exp(offset)./(1+exp(offset)).^2)-1)-offset;
% if isfield(param, 'L_init')
%     L = param.L_init;
%     Y_init = log(Y_0+1)-mean(log(Y_0+1),1);
%     Y = zeros(N,T);
%     for t = 1:T
% %         f = @(y) y + L*log(y) - X_noisy(:,t) + 1;
%         f = @(y) exp(y) + L*y - X_noisy(:,t) + 1;
%         Y(:,t) = fsolve(f, Y_init(:,t));
%         t
%     end
% end
% Y = Y - offset;

% V = ones(N,T)*0;

% if isfield(param, 'L_init')
%     L = param.L_init;
%     Y = zeros(N,T);
%     Y_init = log(Y_0+1)-mean(log(Y_0+1),1);
%     for t = 1:T
%         x = X_noisy(:,t);
%         x(x==0) = 1;
%         y = (diag(x) + L)\(-1 + x.*log(x));
%         Y(:,t) = y;
% %         H = diag(-(x-1)./exp(y).^2) + diag(-1./exp(y).^2) * diag(L*y) + diag(1./exp(y))*L*diag(1./exp(y));
%     end
% end
% Y = Y - mean(Y,1);
% Y = Y - offset;

% L_iter = zeros(N,N,10);

for i = 1:max_iter
    
    % Step 1: given Y, update L
    if (isfield(param, 'L_init') && i==1)
        L = param.L_init;
    else
%         Ys = zeros(N,m,T);
%         for t = 1:T
%             x = X_noisy(:,t);
%             x1 = x;
%             x1(x==0) = 1;
%             y = (diag(x) + L)\(-1 + x.*log(x1));
%             H = diag(-(x-1)./exp(y).^2) + diag(-1./exp(2*y)) * diag(L*y) + diag(1./exp(y))*L*diag(1./exp(y));
%             ys = mvnrnd(y,-inv(H),m);
%             Ys(:,:,t) = ys;
%         end
%         Ys = reshape(Ys,[N,m*T]);
        switch reg_type
            case 'lasso'
                L = graph_learning_logdet_reglap(Ys,param);
            case 'l2'
                Z = gsp_distanz(Ys').^2;
                W = gsp_learn_graph_l2_degrees(Z/beta/4/T,1,struct("maxit",200));
                W(W<0) = 0;
                L = diag(sum(W,1))-W;
            case 'log'
                Z = gsp_distanz(Ys').^2;
                W = gsp_learn_graph_log_degrees(Z/beta/2/T,1,gamma/beta/2);
                W(W<0) = 0;
                L = diag(sum(W,1))-W;
            case 'cgl'
                S = (Y*Y')/T;
                A_mask = ones(size(S))-eye(size(S));
                L = estimate_cgl(S,A_mask,beta,1e-4,1e-6,40,1);
        end
    end
    if any(isnan(L),'all')
        if i == 1
            Y = zeros(N, T);
            O_iter(:,i) = zeros(N, 1);
            L_iter(:,:,i) = zeros(N);
        end
        return
    end
    L_iter(:,:,i) = L;
%     sum(exp(offset+Y+exp(V)/2),'all') - sum(Y_0.*(offset+Y),'all') - sum(V, 'all')/2 + alpha/2*vec(Y*Y')'*vec(L)/T + alpha*beta*trace(L) - alpha/2*log(det(L+1/N))
    
    % Step 3: Update offset
    for j = 1:5
        grad = sum(- Y_0 + 1./(1+exp(-Y-offset)), 2);
        hess = sum(1./(exp(Y+offset)+2+exp(-Y-offset)), 2);
        offset = offset - grad./hess;
    end
    O_iter(:,i) = offset;

    % Step 2: Given L, update Y
    for j = 1:1
        grad = - Y_0 + 1./(1+exp(-Y-offset)) + alpha/T * L * Y;
        hess = alpha/T * L + (eye(N).*repmat(reshape(1./(exp(Y+offset)+2+exp(-Y-offset)),[1,N,T]),[N,1,1]));
        grad_ct = [grad; zeros(1,T)];
        hess_ct = ones(N+1,N+1,T);
        hess_ct(1:N,1:N,:) = hess;
        hess_ct(end,end,:) = 0;
        D = -pagemldivide(hess_ct,reshape(grad_ct,[N+1,1,T]));
        t = 0.5;
        Y = Y+t*squeeze(D(1:N,:));
    end

%     % Step 2: Given L, update V
%     for j = 1:1
%         grad = (-1./V + exp(offset+Y+V/2) + diag(L)) / 2;
%         hess = eye(N).*repmat(reshape((1./V.^2 + exp(offset+Y+V/2)/2)/2, [1,N,T]),[N,1,1]);
%         D = -pagemldivide(hess,reshape(grad,[N,1,T]));
%         t = 0.5;
%         V = V+t*squeeze(D);
%         V(V<0) = 1e-4;
%     end

%     % plot the objective
%     switch reg_type
%         case 'l2'
%             objective(i) = sum(exp(offset+Y+V/2),'all') - sum(Y_0.*(offset+Y),'all') + alpha/2*vec(Y*Y')'*vec(L)/T + alpha/2*beta*(norm(L,'fro')^2);% + rho/2 * norm(Y,'fro')^2;
%         case 'log'
%             objective(i) = sum(exp(offset+Y+V/2),'all') - sum(Y_0.*(offset+Y),'all') + alpha/2*vec(Y*Y')'*vec(L)/T - alpha/2*gamma*sum(log(diag(L))) + alpha/2*beta*(norm(L-diag(diag(L)),'fro')^2);
%         case 'cgl'
%             objective(i) = sum(exp(offset+Y+V/2),'all') - sum(Y_0.*(offset+Y),'all') + alpha/2*vec(Y*Y')'*vec(L)/T + alpha*beta*trace(L) - alpha/2*log(det(L+1/N));% + rho/2 * norm(Y,'fro')^2;
%     end
%     
%     sum(exp(offset+Y+exp(V)/2),'all') - sum(Y_0.*(offset+Y),'all') - sum(V, 'all')/2 + alpha/2*vec(Y*Y')'*vec(L)/T + alpha*beta*trace(L) - alpha/2*log(det(L+1/N))
%     figure(4)
%     plot(i,objective(i), '.r');
%     hold on, drawnow
    
    % stopping criteria
    if i == max_iter
        return
    end
%     if i>=2 && abs(objective(i)-objective(i-1))/abs(objective(i-1))<10^(-6)
%         return
%     end

end
end