nreplicate = 20; % repeat the same experiment (based on different graphs)
for ii = 1:nreplicate

param.N = 20;
[A,XCoords, YCoords] = construct_graph(param.N,'gaussian',0.1,0.2);
% while 1
%     [A,XCoords, YCoords] = construct_graph(param.N,'gaussian',0.6,0.2);
% %     [A,XCoords, YCoords] = construct_graph(param.N,'gaussian',0.75,0.5);
% %     [A,XCoords, YCoords] = construct_graph(param.N,'er',0.2);
% %     [A,XCoords, YCoords] = construct_graph(param.N,'pa',1);
%     % [A,XCoords, YCoords] = construct_graph(param.N,'sbm',0.4,0.1);
%     
%     % [A,XCoords, YCoords] = construct_graph(param.N,'ws',2,0.1);
% 
%     if all(sum(A,2))
%         break
%     end
% end


% A = data{ii,1};
L_0 = full(sgwt_laplacian(A,'opt','raw'));
L_0 = L_0/trace(L_0)*param.N;

% % L_0 = data{ii,2};
% A_0 = -L_0+diag(diag(L_0));
% W = triu(rand(param.N)*1.9 + 0.1, 1);
% A_0 = A_0 .* (W+W');
% L_0 = diag(sum(A_0)) - A_0;
% L_0 = L_0/trace(L_0)*param.N;

param.scale = 1;
param.offset = 0;%unifrnd(-2,4,[1,2000]);

[V,D] = eig(full(L_0));
sigma = pinv(D);
mu = zeros(1,param.N);
num_of_signal = 2000;%+2;
% param.T = num_of_signal;
gftcoeff = mvnrnd(mu,sigma,num_of_signal);
X = V*gftcoeff'*param.scale+param.offset;


% X = data{ii,3};
% X_ = [X(:,1),X,X(:,end)];
% X = (X_(:,1:end-2) + X_(:,2:end-1) + X_(:,3:end))/3;
% X = X - param.offset;

offset = [2*ones(param.N/2,1);-2*ones(param.N/2,1)];
% offset = [-1*ones(param.N/2,1);-2*ones(param.N/2,1)]; % binomial

X_noisy = poissrnd(exp(X+offset), size(X));
% X_log = log(X_noisy+1);
% X_noisy = zeros(param.N,num_of_signal,10);
% for j = 1:10
%     X_noisy(:,:,j) = poissrnd(exp(X+offset), size(X));
% end

% X_noisy = binornd(1,1./(1+exp(-X-offset)));

% r = 2;
% X_noisy = nbinrnd(r,r./(exp(X)+r));

% offset = unifrnd(1,5,[1,1,10]);
% [V,D] = eig(full(L_0));
% sigma = pinv(D);
% mu = zeros(1,param.N);
% X = zeros(param.N,200,10);
% for k = 1:10
%     gftcoeff = mvnrnd(mu,sigma,200);
%     X(:,:,k) = V*gftcoeff';
% end
% X = X + offset;
% X_noisy = poissrnd(exp(X), size(X));
% X_log = log(X_noisy+1);

data{ii,1} = A;
data{ii,2} = L_0;
data{ii,3} = X;
data{ii,4} = X_noisy;

end

%% 
% for ii = 1:nreplicate
%     subplot(4,5,ii);
%     imagesc(pinv(data{ii,2}));
%     colorbar();
% %     imagesc(data{ii,4});
% end
% 
% figure(1)
for i = 1:20
    subplot(4,5,i);
    imagesc(data{i,2});
    sum(data{i,2}<0,'all')
end