
rng(0,'twister');


%read train data

Data = load('umist_cropped.mat');


X=[];%training set
labels  = [];
for i=1:20
    for j=1:(size(Data.facedat{i},3)-14)
        labels = [labels; i;];
        img=Data.facedat{1,i}(:,:,j);
        %img=rgb2gray(img);
        if size(img,3) > 1
            img = rgb2gray(img);
        end
        [irow, icol] = size(img);
        img2 = im2double(img); 
        temp = reshape(img2',irow * icol , 1);
        X = [X temp];% 'T' grows after each turn
    end
end




uniqlabels = unique(labels);

% finding number of unique classes
c = max(size(uniqlabels));


% m = dimensionality of training data
% n = total no of training samples
[m, n] = size(X);

%read test data
TestData=[];%training set
TestLabels = [];
for i=1:20
    for j=(size(Data.facedat{i},3)-13):(size(Data.facedat{i},3))
        TestLabels = [TestLabels; i;];
        img=Data.facedat{1,i}(:,:,j);
        if size(img,3)>1
            img=rgb2gray(img);
        end 
        [irow, icol] = size(img);
        img2 = im2double(img);
        temp = reshape(img2',irow * icol , 1);
        TestData = [TestData temp];% 'T' grows after each turn
    end
end

clear i j irow icol img str temp img2
testdata_n = size(TestData,2);
Predictions = zeros(testdata_n,1);


% noise threshold for data
epsilon = 0.0001;

error_count = 0;

%define vector to save scores for each class
scores = zeros(testdata_n,c);


%{

% used for calculating gamma for RBF kernel
mean_x = mean(X,2);
gamma = median (norm((X - mean_x),2).^(-2))

%calculating RBF gram matrix
n1sq = sum(X.^2,1);
n1 = size(X,2);
temp = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
K = exp(temp.*-gamma);


%}

%Using Linear Kernel
K = X'*X;

%{

%Finding Pseudo transformation matrix using KPCA
%Finding Eigen vectors and Eigen values
[V,D] = eig(K);
if ~issorted(diag(D), 'descend')
    [V,D] = eig(K);
    [D,I] = sort(diag(D),'descend');
    V = V(:, I);
end

%Normalizing eigen vectors
D1 = D.*sqrt(D);
D = D./D1;
V = V*diag(D);

%Select the first 10 eigen vectors for B
B = V(:,1:140);

%}

B = rand(295,40);





for j = 1:testdata_n
    test = TestData(:,j);

    %{
    %calculating RBF test
    n2sq = sum(test.^2,1);
    n2 = size(test,2);
    temp = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*test;
    k = exp(temp.*-gamma);
    %}
    
    %Using Linear Kernel
    k = X'*test;


    cvx_begin
      cvx_quiet(true);
      %coefficient vector to be found
      variable a(n,1);
      minimize norm(a,1);
      subject to
        norm(B'*k - B'*K*a, 2) <= epsilon   
    cvx_end




    %calculate residuals and scores
    for i=1:c
        delta_i = zeros(n,1);
        delta_i(find(labels==uniqlabels(i)),1) = a(find(labels==uniqlabels(i)),1);
        Residual_i = B'*k  - B'*K*delta_i;
        scores(j,i) = norm(Residual_i,2);
    end 
    [minval , index] = min(scores(j,:));
    Predictions(j,1) = uniqlabels (index);
    if (Predictions(j,1) ~= TestLabels(j,1))
        error_count = error_count +1;
        fprintf('Should be %f, but was %f.\n',TestLabels(j,1),Predictions(j,1));
    end
    
    
end

immse(Predictions,TestLabels)
error_count


%{
%src with noise tolerance
%computations for coefficient vector using cvx
cvx_begin
  %cvx_quiet(true);
  %coefficient vector to be found
  variable a(n,1);
  minimize norm(a,1);
  subject to
    norm(test - X*a, 2) <= epsilon   
cvx_end

%{
for i=1:c
    R=test-a()*Traindata(find(Trainlabels==uniqlabels(i)),:);
    src_scores(:,i)=sqrt(sum(R.*R,2));
end
%}

Residual_1 =  test - X(:,n1)*a(n1,1)
score1 = sqrt(sum(Residual_1.*Residual_1,2))

Residual_2 =  test - X(:,n2)*a(n2,1)
score2 = sqrt(sum(Residual_2.*Residual_2,2))


%}

%[predictions,src_scores]=src(X,labels,Y,0.3)