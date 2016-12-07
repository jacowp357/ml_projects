function sepsetBeliefs = calculateSepsetBeliefs(cluster)
%CALCULATESEPSETBELIEFS Determines all the sepset beliefs from the cluster
% sepsetBeliefs = calculateSepsetBeliefs(cluster)
% An array is returned giving the potentials associated with each sepset,
% the indices does not signify anything.

sepsetBeliefs = [];
for i = 1:length(cluster)
    for k = 1:length(cluster(i).sepset)
        
        % Clusters associated with current sepset
        if isempty(cluster(i).sepset{k})
            continue;
        end
        clusters = [i, k];
        alreadyAdded = false;
        
        % Check if already added
        for j = 1:length(sepsetBeliefs)
            if isempty(setdiff(clusters, sepsetBeliefs(j).clusters))
                % Already added
                alreadyAdded = true;
                break;
            end
        end
        if alreadyAdded
            continue;
        end
        
        % If not, create and add new sepset belief potential
        newSepset = normpot(multpots([cluster(i).message(k), cluster(k).message(i)]));
        newSepset.clusters = clusters;
        sepsetBeliefs = [sepsetBeliefs, newSepset];
        
    end
end
