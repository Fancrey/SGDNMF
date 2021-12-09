function [C,Newdata, Newgnd] = solve_C(data, label, testRatio, nClass)
% testRatio: Proportion of test data in total data
% label: the label of all data
% data: m*n m:Feature dimension. n: Number of data
% 训练集索引
testRatio=1-testRatio;
n=size(data, 2);
m=size(data, 1);
if testRatio~=1
trainIndices = crossvalind('HoldOut', n, testRatio);
%trainIndices = logical(floor(rand(n,1)*(1/testRatio)));
% 测试集索引
testIndices = ~trainIndices;

% 训练集和训练标签
trainData = data(:,trainIndices);
trainLabel = label(trainIndices,:);
k=size(trainData,2);
length(unique(trainLabel))
% 测试集和测试标签
testData = data(:,testIndices);
testLabel= label(testIndices, :);


Newdata=[trainData testData];
Newgnd=[trainLabel;testLabel];
C = zeros(n,n+nClass-k);
Q=zeros(k,nClass);
for i=1:k
Q(i,trainLabel(i))=1;
end
C=[Q zeros(k,n-k);zeros(n-k,nClass) eye(n-k)];
else
    k=0
    C = zeros(n,n+nClass-k);
    Q=zeros(k,nClass);
    C=[Q zeros(k,n-k);zeros(n-k,nClass) eye(n-k)];
    Newdata=data;
    Newgnd=label;
end
    
