function newCluster = passMessage(cluster, i, j)
%PASSMESSAGE Pass a message from one cluster (i) to another cluster (j)
% newCluster = passMessage(cluster, fromCluster, toCluster)

% Initialise new clusters
newCluster = cluster;
ci = newCluster(i); % going from cluster i

% Calculate the product of the messages coming into i excluding j's message
productPots = [];
for k = 1:length(ci.message)
    if (isempty(ci.message(k).variables)) | (k == j)
        continue
    end
    productPots = [productPots, ci.message(k)];
end

if isempty(productPots)
    product = ci.pot;
else
    product = productPots; % the product of the incoming messages to cluster i
    product = multpots([product, ci.pot]);
end

% Calculate the message
newCluster(j).message(i) = normpot(sumpot(product, ci.sepset{j}, 0));
