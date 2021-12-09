function [C,Newdata, Newgnd] = solve_C(data, label, testRatio, nClass)
% testRatio: Proportion of test data in total data
% label: the label of all data
% data: m*n m:Feature dimension. n: Number of data
%?
testRatio=1-testRatio;
n=size(data, 2);
m=size(data, 1);
if testRatio~=1
    trainIndices = crossvalind('HoldOut', n, testRatio);
    trainLabel = label(trainIndices,:);
    if length(unique(trainLabel))<length(unique(label))
        qs=find(ismember(unique(label),unique(trainLabel))==0);
        if length(qs)==1
            candid=find(qs==label);
            trainIndices(candid(1))=1;
        else
            qs_num=length(qs);
            for index=1:qs_num
                candid=find(qs(index)==label);
                trainIndices(candid(1))=1;
            end
        end
    end
    trainLabel = label(trainIndices,:);
    testIndices = ~trainIndices;
    trainData = data(:,trainIndices);
    k=size(trainData,2);
    length(unique(trainLabel))
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

