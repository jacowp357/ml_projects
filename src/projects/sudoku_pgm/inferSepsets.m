function newCluster = inferSepsets(cluster)
%INFERSEPSETS Makes sure that all the sepsets are consistent
% newCluster = inferSepsets(cluster)

newCluster = cluster;

for k = 1:length(cluster)
    if length(newCluster(k).sepset) < length(cluster)
        newCluster(k).sepset{length(newCluster)} = []; % has to have as many possible sepsets as there are cluster
    end
    for j = 1:length(newCluster)
        if ~isempty(newCluster(k).sepset{j})
            newCluster(j).sepset{k} = newCluster(k).sepset{j};
        end
    end
end
