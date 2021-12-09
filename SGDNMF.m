function [Z ,C, A, outputs]= SGDNMF(X,layers,label,options)
%%%%%%%%%%%%%%%%%%%%%%%%
% This Code is to solve the following problem
%%
% $min ||X- Z_1 Z_2...Z_L(C A_L)^T||_F^2+\sum_{i=1}{L}\alpha_i Tr((C A_i^T)^T L_i^H C A_i+ Z_i^T L_i^Z Z_i)$
% s.t. Z_i^TZ_i=I,A_i^TA_i=I A_i >=0
%%%%%%%%%%%%%%%%%%%%%%%
%           X : m*n Data Matrix
%      layers : Dimensions of each layer example: [1024,512,256]
%       label : a n*1 vector, The data points of known label is labeled as class_number, and the unknown is 0
%     options : options.labelRatio : Known data ratio
%               options.maxiter : maxiter number
%               options.alpha   : dual-hypergraph parameter
%               options.beta    : orth parameter
%               options.WeightMode : 'Binary'    'HeatKernel'   'Cosine'
%               options.verbose : 0-1
%   The code is created by Haonan Wu 
%   2021.12.09
%   e-mail:fancrey@gmail.com
%   reference:
%   Y. Meng, R. Shang, F. Shang, L. Jiao, S. Yang and R. Stolkin, “Semi-Supervised Graph Regularized Deep NMF 
%   With Bi-Orthogonal Constraints for Data Representation,” IEEE Transactions on Neural Networks and 
%   Learning Systems, vol. 31, no. 9, pp. 3245-3258, 2020.
X = (NormalizeFea(X'))';
options.bNormalized = 1;
labelRatio = options.labelRatio;
maxiter    = options.maxiter;
alpha      = options.alpha;
beta       = options.beta;
if ~isfield(options,'verbose')
    verbose=0;
else
    verbose=options.verbose;
end
%% Initializing C_L according to known_ Label
nClass=length(unique(label));
if isfield(options,'C')
    C=options.C;
    outputs.gnd=label;
else
    [C, X, gnd] = solve_C(X, label, labelRatio, nClass);
    outputs.gnd=gnd;
end


%% Initializing Layers
n_layer=length(layers);
Z = cell(1, n_layer);
A = cell(1, n_layer);
Wz =cell(1, n_layer);
Wh =cell(1, n_layer);
Lh =cell(1, n_layer);
Lz =cell(1, n_layer);
dnormarray = zeros(2,maxiter);
dnorm=zeros(1,maxiter);
for i_layer=1:n_layer
    if i_layer == 1
        Tmp = X;
    else
        Tmp = (C*A{i_layer - 1})';
    end
    fprintf('Initialising Layer #%d ...\n', i_layer);
    [Z{i_layer}, A{i_layer}] = ConstraintNMF(Tmp,C,layers(i_layer)); %Initializing by CNMF
    Z{i_layer} = normalize_W(Z{i_layer}, 2);
    A{i_layer} = (normalize_H(A{i_layer}', 2))';
    disp('Finishing initialization ...');
end
clear Tmp
disp('Fine-tuning ...');


%% Training Layers
% \Psi -> S
%XXT=constructKernel(X,[]);
for iter = 1:maxiter
    %%%%%%%%%
   
    
    for i=1:n_layer
        %%%%%%%%%% Upadate Z_i
        if i == 1
            Wz1 = constructW(X,options);
            DCol = full(sum(Wz1,2));
            Dz = spdiags(DCol,0,speye(size(Wz1,1)));
            Lz{1} = Dz - Wz1;
            ZACCA=Z{i}*A{i}'*(C'*C)*A{i};
            DZ=alpha*Dz*Z{i};
            XCA=X*C*A{i}+alpha*Wz1*Z{i};
            Z{i}=Z{i}.*(XCA)./(ZACCA+DZ+beta*Z{i});
        else
            Wz{i} = constructW(Z{i-1}',options);
            DCol = full(sum(Wz{i},2));
            Dz = spdiags(DCol,0,speye(size(Wz{i},1)));
            Lz{i} = Dz - Wz{i};
            SXCA=S'*X*C*A{i}+alpha*Wz{i}*Z{i};
            SSZACCA=S'*S*Z{i}*A{i}'*(C'*C)*A{i};
            DZ=alpha*Dz*Z{i};
            Z{i}=Z{i}.*(SXCA)./(SSZACCA+DZ+beta*Z{i});
        end
        
        %%%%%%%%%% Upadate A_i
        if i==1
            S=Z{i};
            Wh1 = constructW(X',options);
            DCol = full(sum(Wh1,2));
            Dh = spdiags(DCol,0,speye(size(Wh1,1)));
            Lh{1} = Dh - Wh1;
            XS=X'*S;XSpos=(abs(XS)+XS)/2;XSneg=(abs(XS)-XS)/2;
            SS=S'*S;SSpos=(abs(SS)+SS)/2;SSneg=(abs(SS)-SS)/2;
            FZ1=C'*XSpos;FZ2=C'*C*A{i}*SSneg;FZ3=alpha*C'*Wh1*C*A{i};
            FM1=C'*XSneg;FM2=C'*C*A{i}*SSpos;FM3=alpha*C'*Dh*C*A{i}+beta*A{i};
            A{i}=A{i}.*(FZ1+FZ2+FZ3)./(FM1+FM2+FM3);
        else
            S=S*Z{i};
            Wh{i} = constructW(X',options);
            DCol = full(sum(Wh{i},2));
            Dh = spdiags(DCol,0,speye(size(Wh{i},1)));
            Lh{i} = Dh - Wh{i};
            XS=X'*S;XSpos=(abs(XS)+XS)/2;XSneg=(abs(XS)-XS)/2;
            SS=S'*S;SSpos=(abs(SS)+SS)/2;SSneg=(abs(SS)-SS)/2;
            FZ1=C'*XSpos;FZ2=C'*C*A{i}*SSneg;FZ3=alpha*C'*Wh{i}*C*A{i};
            FM1=C'*XSneg;FM2=C'*C*A{i}*SSpos;FM3=alpha*C'*Dh*C*A{i}+beta*A{i};
            A{i}=A{i}.*(FZ1+FZ2+FZ3)./(FM1+FM2+FM3);
        end
        %%%%%%%%%%
    end
    [dnorm1,dnorm2,dnorm3] = cost_function(X, S, C, A, Z, Lh, Lz, alpha);
    dnorm(iter)=dnorm3;
    dnormarray(:,iter) = [dnorm1;dnorm2];
    if verbose
        fprintf('#%d error1: %f error2: %f error: %f \n', iter, dnorm1,dnorm2,dnorm3);
        plot(dnorm(1:iter),'Linewidth',2);
        drawnow
    end
end
outputs.dnorm=dnormarray;
outputs.X=X;

end
function [error1, error2, error] = cost_function(X, S, C, A, Z, Lh, Lz, alpha)
error1 = norm(X - S*A{end}'*C', 'fro');
num=numel(A);
error2=0;
for i=1:num
    error2 = error2 + alpha*trace(A{i}'*C'*Lh{i}*C*A{i}+Z{i}'*Lz{i}*Z{i});
end
error=error1+error2;
end

