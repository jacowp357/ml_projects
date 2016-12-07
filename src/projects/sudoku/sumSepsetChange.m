function change = sumSepsetChange(prevSepset, curSepset)
%SUMSEPSETCHANGE Determines the sum of the absolute change in sepset beliefs
% change = sumSepsetChange(prevSepset, curSepset)

change = 0;
for k = 1:length(prevSepset)
    change = change + sum(abs((prevSepset(k).table - curSepset(k).table)));
end
