function [wo, wb] = makeGMM(A, bgPix, fgPix)

if ~exist('EM.m','file')
    addpath('../em/GMM-GMR-v2.0');
end
addpath('../em/netlib');

A = reshape(A, [], 3)';
ncenters = 5;

X = A(:,bgPix);
[Unaries, Mu, Sigma] = EM_init_kmeans(X, ncenters);
[unarypotb, muB, sigmaB] = EM(X, Unaries, Mu, Sigma);

X = A(:,fgPix);
[Unaries, Mu, Sigma] = EM_init_kmeans(X, ncenters);
[unarypotF, muF, sigmaF] = EM(X, Unaries, Mu, Sigma);

n = size(A,2);

stabF = 0.01;
stabB = 0.01;

wo = zeros(n,ncenters);
wb = wo;

for i=1:ncenters
    L = chol(inv(sigmaB(:,:,i) + stabB*eye(3)));
    tmp = L*(A - repmat(muB(:,i),1,n));
    mean( sum( (A - repmat(muB(:,i),1,n)).^2, 1 ))
    wo(:,i) = -log(unarypotb(i)) + 0.5*log(det(sigmaB(:,:,i) + stabB*eye(3))) + 0.5* sum( tmp.^2, 1)'; 

    L = chol(inv(sigmaF(:,:,i)+stabF*eye(3)));
    tmp = L*(A - repmat(muF(:,i),1,n));
    wb(:,i) = -log(unarypotF(i)) + 0.5*log(det(sigmaF(:,:,i) + stabF*eye(3))) + 0.5* sum( tmp.^2, 1)'; 
end

wo2 = min(wo, [], 2) + 3/2 * log(2*pi);
wb2 = min(wb, [], 2) + 3/2 * log(2*pi);

while min(wo2) < 0
    stabB = stabB + 0.1;
    for i=1:ncenters
        L = chol(inv(sigmaB(:,:,i) + stabB*eye(3)));
        tmp = L*(A - repmat(muB(:,i),1,n));
        wo(:,i) = -log(unarypotb(i)) + 0.5*log(det(sigmaB(:,:,i) + stabB*eye(3))) + 0.5* sum( tmp.^2, 1)';
    end
    wo2 = min(wo, [], 2) + 3/2 * log(2*pi);
end
while min(wb2) < 0
    stabF = stabF + 0.1;
    for i=1:ncenters
        L = chol(inv(sigmaF(:,:,i)+stabF*eye(3)));
        tmp = L*(A - repmat(muF(:,i),1,n));
        wb(:,i) = -log(unarypotF(i)) + 0.5*log(det(sigmaF(:,:,i) + stabF*eye(3))) + 0.5* sum( tmp.^2, 1)';
    end
    wb2 = min(wb, [], 2) + 3/2 * log(2*pi);        
end
wo = wo2;
wb = wb2;
