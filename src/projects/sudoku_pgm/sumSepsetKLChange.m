function change = sumSepsetKLChange(prevSepset, curSepset)
%SUMSEPSETCHANGE Determines the sum of the absolute KL change in sepset beliefs
% change = sumSepsetChange(prevSepset, curSepset)

change = 0;
for k = 1:length(prevSepset)
    change = change + sum(abs((KLDiv(prevSepset(k).table, curSepset(k).table) + KLDiv(prevSepset(k).table, curSepset(k).table))/2));
end
