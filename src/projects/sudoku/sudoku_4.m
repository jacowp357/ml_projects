% Sudoku 4x4 solver %
% J. du Toit 19/11/2015 %

% Initial setup %
close all;
clear all;
clc;

% Constants %
SEPSET_SUM_CONVERGENCE_THRES = 0.00001;
MAX_ITER = 20;

% States %
one = 1; two = 2; three = 3; four = 4;
[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16] = assign([1:32]);
% Setup the variable potentials %
for v = v1:v16
    variable(v).name = strcat('v', num2str(v)); variable(v).domain = {'1', '2', '3', '4'};
end;
for r = r1:r16
    variable(r).name = strcat('r', num2str(r)); variable(r).domain = {'1', '2', '3', '4'};
end;

% Setup the factor connections %
[fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8, fc9, fc10, fc11, fc12] = assign([1:12]);

g = reshape(1:16, 4, 4)';

% Horizontal rules/factors %
i = 1;
for f = fc1:fc4
    pot(f).variables = [g(i, :)];
    i = i + 1;
end;

% Vertical rules/factors %
i = 1;
for f = fc5:fc8
    pot(f).variables = [g(:, i)'];
    i = i + 1;
end;

C = mat2cell(g, [2 2], [2 2])';

% Bricks rules/factors %
i = 1;
for f = fc9:fc12
    pot(f).variables = [reshape(C{i}, 1, 4)];
    i = i + 1;
end;

% Setup the factor rules %
for x1 = one:four
    for x2 = one:four
        for x3 = one:four
            for x4 = one:four
                if length(unique([x1, x2, x3, x4])) == 4
                    for fc = fc1:fc12
                        pot(fc).table(x1, x2, x3, x4) = 1;
                    end;
                else
                    for fc = fc1:fc12
                        pot(fc).table(x1, x2, x3, x4) = 0;
                    end;
                end;
            end;
        end;
    end;
end;

% Setup the fr potentials %
[fr1, fr2, fr3, fr4, fr5, fr6, fr7, fr8, fr9, fr10, fr11, fr12, fr13, fr14, fr15, fr16] = assign([13:28]);
for i = fr1:fr16
    pot(i).variables = [i + 4, i - 12]; % pot(fr1).variables = [r1, b1];
end

for fr = fr1:fr16
    for x1 = one:four
        for x2 = one:four
            if x1 == x2
                pot(fr).table(x1, x2) = 1;
            else
                pot(fr).table(x1, x2) = 0;
            end;
        end;
    end;
end;

% Setup the evidence %
evidence_cells = [4, 5, 10, 15];
%pot(fr1) = setpot(pot(fr1), r1, three);
%pot(fr2) = setpot(pot(fr2), r2, two);
%pot(fr3) = setpot(pot(fr3), r3, one);
pot(fr4) = setpot(pot(fr4), r4, four);
pot(fr5) = setpot(pot(fr5), r5, two);
%pot(fr6) = setpot(pot(fr6), r6, four);
%pot(fr7) = setpot(pot(fr7), r7, one);
%pot(fr8) = setpot(pot(fr8), r8, four);
%pot(fr9) = setpot(pot(fr9), r9, one);
pot(fr10) = setpot(pot(fr10), r10, one);
%pot(fr11) = setpot(pot(fr11), r11, two);
%pot(fr12) = setpot(pot(fr12), r12, four);
%pot(fr13) = setpot(pot(fr13), r13, one);
%pot(fr14) = setpot(pot(fr14), r14, two);
pot(fr15) = setpot(pot(fr15), r15, one);
%pot(fr16) = setpot(pot(fr16), r16, one);


%TODO: graph coloring function here %


% Setup the cluster graph %
% the sepset between C1 and C17 is v1 %

% clusters assignment for all cells involving horizontal and vertical rules %
fr = 13;
for c = 1:16
   for p = 1:12
      if ismember(c, pot(p).variables)
         cluster(c).pot = pot(fr);
         cluster(c).sepset{p + 16} = c;
      end;
   end;
   fr = fr + 1;
end;

% cluster assignment for all cells involving brick rules %
for p = 1:12
   cluster(p + 16).pot = pot(p);
end;

cluster = inferSepsets(cluster); % a sepset only has to be defined once, then it is added automatically by this function

% Koller Slide 8.1.9: Intialise all messages to 1
cluster = initMessages(cluster, 4); % cluster(i).message(j) is the message to this cluster from cluster j

cluster = calculateBeliefs(cluster); % initialise beliefs cluster(i).belief
sepsetBeliefs = calculateSepsetBeliefs(cluster); % calculate sepset beliefs

% these messages do not change, so we can pass them in only once %
for i = 1:length(evidence_cells)
    for p = 1:12
        if ismember(evidence_cells(i), pot(p).variables)
            cluster = passMessage(cluster, evidence_cells(i), p + 16);
        end;
    end;
end;

% Koller Slide 8.1.9: Repeat %
fprintf('iterating: \n');
sumSepsetChanges = [];
for k = 1:MAX_ITER
    % Store previous clusters and sepset beliefs %
    prevCluster = cluster;
    prevSepsetBeliefs = sepsetBeliefs;

    % Koller Slide 8.1.9: Select edge (i, j) and pass message %

    for i = 1:28
        for p = 1:12
            if ismember(i, pot(p).variables)
                if ismember(i, evidence_cells)
                    cluster = passMessage(cluster, p + 16, i);
                    %fprintf('%d -> %d\n', p + 16, i);
                else
                    cluster = passMessage(cluster, i, p + 16);
                    cluster = passMessage(cluster, p + 16, i);
                    %fprintf('%d -> %d\n', i, p + 16);
                end;
            end;
        end;
    end;

%     cluster = passMessage(cluster, 1, 17);
%     cluster = passMessage(cluster, 17, 1);
%     cluster = passMessage(cluster, 1, 21);
%     cluster = passMessage(cluster, 21, 1);
%     cluster = passMessage(cluster, 1, 25);
%     cluster = passMessage(cluster, 25, 1);
%
%     cluster = passMessage(cluster, 2, 17);
%     cluster = passMessage(cluster, 17, 2);
%     cluster = passMessage(cluster, 2, 22);
%     cluster = passMessage(cluster, 22, 2);
%     cluster = passMessage(cluster, 2, 25);
%     cluster = passMessage(cluster, 25, 2);
%
%     cluster = passMessage(cluster, 3, 17);
%     cluster = passMessage(cluster, 17, 3);
%     cluster = passMessage(cluster, 3, 23);
%     cluster = passMessage(cluster, 23, 3);
%     cluster = passMessage(cluster, 3, 26);
%     cluster = passMessage(cluster, 26, 3);
%
% %     cluster = passMessage(cluster, 4, 17);
%     cluster = passMessage(cluster, 17, 4);
% %     cluster = passMessage(cluster, 4, 24);
%     cluster = passMessage(cluster, 24, 4);
% %     cluster = passMessage(cluster, 4, 26);
%     cluster = passMessage(cluster, 26, 4);
%
% %     cluster = passMessage(cluster, 5, 18);
%     cluster = passMessage(cluster, 18, 5);
% %     cluster = passMessage(cluster, 5, 21);
%     cluster = passMessage(cluster, 21, 5);
% %     cluster = passMessage(cluster, 5, 25);
%     cluster = passMessage(cluster, 25, 5);
%
%     cluster = passMessage(cluster, 6, 18);
%     cluster = passMessage(cluster, 18, 6);
%     cluster = passMessage(cluster, 6, 22);
%     cluster = passMessage(cluster, 22, 6);
%     cluster = passMessage(cluster, 6, 25);
%     cluster = passMessage(cluster, 25, 6);
%
%     cluster = passMessage(cluster, 7, 18);
%     cluster = passMessage(cluster, 18, 7);
%     cluster = passMessage(cluster, 7, 23);
%     cluster = passMessage(cluster, 23, 7);
%     cluster = passMessage(cluster, 7, 26);
%     cluster = passMessage(cluster, 26, 7);
%
%     cluster = passMessage(cluster, 8, 18);
%     cluster = passMessage(cluster, 18, 8);
%     cluster = passMessage(cluster, 8, 24);
%     cluster = passMessage(cluster, 24, 8);
%     cluster = passMessage(cluster, 8, 26);
%     cluster = passMessage(cluster, 26, 8);
%
%     cluster = passMessage(cluster, 9, 19);
%     cluster = passMessage(cluster, 19, 9);
%     cluster = passMessage(cluster, 9, 21);
%     cluster = passMessage(cluster, 21, 9);
%     cluster = passMessage(cluster, 9, 27);
%     cluster = passMessage(cluster, 27, 9);
%
% %     cluster = passMessage(cluster, 10, 19);
%     cluster = passMessage(cluster, 19, 10);
% %     cluster = passMessage(cluster, 10, 22);
%     cluster = passMessage(cluster, 22, 10);
% %     cluster = passMessage(cluster, 10, 27);
%     cluster = passMessage(cluster, 27, 10);
%
%     cluster = passMessage(cluster, 11, 19);
%     cluster = passMessage(cluster, 19, 11);
%     cluster = passMessage(cluster, 11, 23);
%     cluster = passMessage(cluster, 23, 11);
%     cluster = passMessage(cluster, 11, 28);
%     cluster = passMessage(cluster, 28, 11);
%
%     cluster = passMessage(cluster, 12, 19);
%     cluster = passMessage(cluster, 19, 12);
%     cluster = passMessage(cluster, 12, 24);
%     cluster = passMessage(cluster, 24, 12);
%     cluster = passMessage(cluster, 12, 28);
%     cluster = passMessage(cluster, 28, 12);
%
%     cluster = passMessage(cluster, 13, 20);
%     cluster = passMessage(cluster, 20, 13);
%     cluster = passMessage(cluster, 13, 21);
%     cluster = passMessage(cluster, 21, 13);
%     cluster = passMessage(cluster, 13, 27);
%     cluster = passMessage(cluster, 27, 13);
%
%     cluster = passMessage(cluster, 14, 20);
%     cluster = passMessage(cluster, 20, 14);
%     cluster = passMessage(cluster, 14, 22);
%     cluster = passMessage(cluster, 22, 14);
%     cluster = passMessage(cluster, 14, 27);
%     cluster = passMessage(cluster, 27, 14);
%
% %     cluster = passMessage(cluster, 15, 20);
%     cluster = passMessage(cluster, 20, 15);
% %     cluster = passMessage(cluster, 15, 23);
%     cluster = passMessage(cluster, 23, 15);
% %     cluster = passMessage(cluster, 15, 28);
%     cluster = passMessage(cluster, 28, 15);
%
%     cluster = passMessage(cluster, 16, 20);
%     cluster = passMessage(cluster, 20, 16);
%     cluster = passMessage(cluster, 16, 24);
%     cluster = passMessage(cluster, 24, 16);
%     cluster = passMessage(cluster, 16, 28);
%     cluster = passMessage(cluster, 28, 16);

    % Koller Slide 8.1.9: Compute beliefs
    cluster = calculateBeliefs(cluster);
    sepsetBeliefs = calculateSepsetBeliefs(cluster);

    % check for Kullback Leibler distance convergence %
    change = sumSepsetKLChange(prevSepsetBeliefs, sepsetBeliefs);
    sumSepsetChanges = [sumSepsetChanges, change];
    if change <= SEPSET_SUM_CONVERGENCE_THRES
        fprintf(' converged (%d iterations)', k)
        break
    end

fprintf('\n');

% Decode by looking at max beliefs %
s = 0;
b = 0;
for i = v1:v16
    [tmp0, tmp1] = maxpot(cluster(i).belief, [], 0);
    s(i) = tmp1(1);
    b(i) = tmp0.table;
end;

clc;
fprintf('| %d (%f) | %d (%f) | %d (%f) | %d (%f) |\n', s(1), b(1), s(2), b(2), s(3), b(3), s(4), b(4));
fprintf('| %d (%f) | %d (%f) | %d (%f) | %d (%f) |\n', s(5), b(5), s(6), b(6), s(7), b(7), s(8), b(8));
fprintf('| %d (%f) | %d (%f) | %d (%f) | %d (%f) |\n', s(9), b(9), s(10), b(10), s(11), b(11), s(12), b(12));
fprintf('| %d (%f) | %d (%f) | %d (%f) | %d (%f) |\n', s(13), b(13), s(14), b(14), s(15), b(15), s(16), b(16));
pause(0.2);

end;


% Plot sepset beliefs sum %
plot(1:length(sumSepsetChanges), sumSepsetChanges);
xlabel('Iterations');
ylabel('Sum(KL distance) in sepset beliefs');
grid on;
