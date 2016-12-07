function newCluster = initMessages(cluster, scope)
%INITMESSAGES Initialise all messages to 1
% newCluster = initMessages(cluster)

newCluster = cluster;

for i = 1:length(newCluster)
    for j = 1:length(newCluster(i).sepset)
        if ~(isempty(newCluster(i).sepset{j}))
            newCluster(i).message(j).variables = newCluster(i).sepset{j};
            onesDim = [];
            for k = 1:length(newCluster(i).sepset{j})
                onesDim = [onesDim, scope];
            end
            onesDim = [onesDim, 1];
            newCluster(i).message(j).table = ones(onesDim);
            newCluster(i).message(j) = normpot(newCluster(i).message(j));
        end
    end
end
