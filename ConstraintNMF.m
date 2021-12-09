function [U,V_final] = ConstraintNMF(X,A,k)


m=size(X,1);
maxIter=5;
nIter=0;




Ub=rand(m,k);
[Ub] = NormalizeK(Ub);
Zb=rand(size(A,2),k);
[Zb] = NormalizeK(Zb);
while nIter<maxIter
    
    % -----update U
    C=X*A*Zb;
    D=Ub*Zb'*A'*A*Zb;
    U=Ub.*(C./max(D,1e-10));
    Ub=U;

    % -----update Z
    E=A'*X'*Ub;
    F=A'*A*Zb*Ub'*Ub;
    Z=Zb.*(E./max(F,1e-10));    
    Zb=Z;

    
     FR=norm(X-Ub*Zb'*A');
    nIter=nIter+1;
%     Qf(nIter,1)=FR;
    
end
V=A*Z;
V_final = Z;





    function [K] = NormalizeK(K)
        n = size(K,2);
            norms = max(1e-15,sqrt(sum(K.^2,1)))';
            K = K*spdiags(norms.^-1,0,n,n);