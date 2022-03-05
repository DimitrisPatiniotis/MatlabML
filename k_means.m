% K-means clustering

% The dataset consists of 150 datapoints of 2 features from iris dataset
%

clear all; close all; clc


X=load('datasets/kmeans.dat'); 
K=3; % number of centroids
n=size(X) % check the data size
max_iterations = 10; % if this number is too low a warning will be displayed stating that the algorithm did not converge, which you should expect since the software only implemented one iteration.

% Alternative way to load data
% load fisheriris
% X = meas(:,3:4)

figure;
plot(X(:,1),X(:,2),'k.','MarkerSize',12);
title 'Iris Data';
xlabel 'Petal Lengths (cm)'; 
ylabel 'Petal Widths (cm)';


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 1. Using 'kmeans' function: "kmeans(x,k)" performs k-means clustering to partition the observations of the n-by-p data matrix X into k clusters, and returns an n-by-1 vector (idx) containing cluster indices of each observation. Rows of X correspond to points and columns correspond to variables. 

rng(1); % For reproducibility
[idx,C] = kmeans(X,K);

x1 = min(X(:,1)):0.01:max(X(:,1));
x2 = min(X(:,2)):0.01:max(X(:,2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot

idx2Region = kmeans(XGrid,K,'MaxIter',max_iterations,'Start',C); 

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12)
plot(C(:,1),C(:,2),'k*','MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Centroids','Location','NW')
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)';
title 'Cluster Assignments and Centroids using built-in function'
hold off





% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 2. Using your own function

%Randomly Initialize Centroids
len = size(X,1);
rand_num = 1+floor((len-1)*rand(K,1));
centroids = X(rand_num,:);

for i=1:max_iterations
    indices = zeros(size(X,1), 1);

    for i=1:length(X)
        % calculate distance from centroid
        distance = pdist2(centroids, X(i,:));
        [C, indices(i)]= min(distance);
    end
    [m, n] = size(X);
    centroids = zeros(K, n);

    for i=1:K
        %Find the new cluster center location using mean
        centroids(i,:) = mean(X(find(idx==i),:)); 
    end

end


figure;
plot(X(indices==1,1),X(indices==1,2),'r.','MarkerSize',12)
hold on
plot(X(indices==2,1),X(indices==2,2),'b.','MarkerSize',12)
plot(X(indices==3,1),X(indices==3,2),'g.','MarkerSize',12)
plot(centroids(:,1),centroids(:,2),'k*','MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Centroids','Location','NW')
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)';
title 'Cluster Assignments and Centroids using your implementation'
hold off