function newCluster = calculateBeliefs(cluster)
%CALCULATEBELIEFS Calculate the beliefs of the clusters
% newCluster = calculateBeliefs(cluster)
% To each cluster in the given sets of clusters a new field "belief" is
% added, a potential giving the beliefs of that cluster.

newCluster = cluster;

for i = 1:length(newCluster)
    productPots = [];
    for k = 1:length(newCluster(i).sepset)
        if ~(isempty(newCluster(i).sepset{k}))
            productPots = [productPots, newCluster(i).message(k)];
        end
    end
    newCluster(i).belief = normpot(multpots([newCluster(i).pot, productPots]));
end
