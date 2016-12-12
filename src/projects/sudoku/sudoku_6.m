% Sudoku 6x6 solver %
% J. du Toit 19/11/2015 %

% Initial setup %
close all;
clear all;
clc;

% Constants %
SEPSET_SUM_CONVERGENCE_THRES = 0.00001;
MAX_ITER = 30;

% States %
one = 1; two = 2; three = 3; four = 4; five = 5; six = 6;
[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36] = assign([1:72]);
% Setup the variable potentials %
for v = v1:v36
    variable(v).name = strcat('v', num2str(v)); variable(v).domain = {'1', '2', '3', '4', '5', '6'};
end;
for r = r1:r36
    variable(r).name = strcat('r', num2str(r)); variable(r).domain = {'1', '2', '3', '4', '5', '6'};
end;
fprintf('Variable potentials set up...\n');

% Setup the factor connections %
[fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8, fc9, fc10, fc11, fc12, fc13, fc14, fc15, fc16, fc17, fc18] = assign([1:18]);
fprintf('Factor connections set up...\n');
g = reshape(1:36, 6, 6)';

% Horizontal rules/factors %
i = 1;
for f = fc1:fc6
    pot(f).variables = [g(i, :)];
    i = i + 1;
end;
fprintf('Horizontal factor scope set up...\n');

% Vertical rules/factors %
i = 1;
for f = fc7:fc12
    pot(f).variables = [g(:, i)'];
    i = i + 1;
end;
fprintf('Vertical factor scope set up...\n');

C = mat2cell(g, [2 2 2], [3 3])';

% Bricks rules/factors %
i = 1;
for f = fc13:fc18
    pot(f).variables = [sort(reshape(C{i}, 1, 6), 'ascend')];
    i = i + 1;
end;
fprintf('Bricks factor scope set up...\n');

% Setup the factor rules %
for x1 = one:six
    for x2 = one:six
        for x3 = one:six
            for x4 = one:six
                for x5 = one:six
                    for x6 = one:six
                        if length(unique([x1, x2, x3, x4, x5, x6])) == 6
                            for fc = fc1:fc18
                                pot(fc).table(x1, x2, x3, x4, x5, x6) = 1;
                            end;
                        else
                            for fc = fc1:fc18
                                pot(fc).table(x1, x2, x3, x4, x5, x6) = 0;
                            end;
                        end;
                    end;
                end;
            end;
        end;
    end;
end;
fprintf('All factor potentials set up...\n');

% Setup the fr potentials %
[fr1, fr2, fr3, fr4, fr5, fr6, fr7, fr8, fr9, fr10, fr11, fr12, fr13, fr14, fr15, fr16, fr17, fr18, fr19, fr20, fr21, fr22, fr23, fr24, fr25, fr26, fr27, fr28, fr29, fr30, fr31, fr32, fr33, fr34, fr35, fr36] = assign([19:54]);
for i = fr1:fr36
    pot(i).variables = [i + 18, i - 18]; % pot(fr1).variables = [r1, v1];
end;
fprintf('Received factor connections set up...\n');

for fr = fr1:fr36
    for x1 = one:six
        for x2 = one:six
            if x1 == x2
                pot(fr).table(x1, x2) = 1;
            else
                pot(fr).table(x1, x2) = 0;
            end;
        end;
    end;
end;
fprintf('Received factor potentials set up...\n');

% Setup the evidence %
evidence_cells = [1, 2, 11, 18, 22, 24, 27, 28, 32, 33, 34];
pot(fr1) = setpot(pot(fr1), r1, four);
pot(fr2) = setpot(pot(fr2), r2, five);
%pot(fr3) = setpot(pot(fr3), r3, one);
%pot(fr4) = setpot(pot(fr4), r4, four);
%pot(fr5) = setpot(pot(fr5), r5, one);
%pot(fr6) = setpot(pot(fr6), r6, one);
%pot(fr7) = setpot(pot(fr7), r7, one);
%pot(fr8) = setpot(pot(fr8), r8, six);
%pot(fr9) = setpot(pot(fr9), r9, one);
%pot(fr10) = setpot(pot(fr10), r10, five);
pot(fr11) = setpot(pot(fr11), r11, three);
%pot(fr12) = setpot(pot(fr12), r12, one);
%pot(fr13) = setpot(pot(fr13), r13, six);
%pot(fr14) = setpot(pot(fr14), r14, six);
%pot(fr15) = setpot(pot(fr15), r15, two);
%pot(fr16) = setpot(pot(fr16), r16, three);
%pot(fr17) = setpot(pot(fr17), r17, one);
pot(fr18) = setpot(pot(fr18), r18, six);
%pot(fr19) = setpot(pot(fr19), r19, three);
%pot(fr20) = setpot(pot(fr20), r20, four);
%pot(fr21) = setpot(pot(fr21), r21, two);
pot(fr22) = setpot(pot(fr22), r22, three);
%pot(fr23) = setpot(pot(fr23), r23, six);
pot(fr24) = setpot(pot(fr24), r24, two);
%pot(fr25) = setpot(pot(fr25), r25, one);
%pot(fr26) = setpot(pot(fr26), r26, one);
pot(fr27) = setpot(pot(fr27), r27, one);
pot(fr28) = setpot(pot(fr28), r28, six);
%pot(fr29) = setpot(pot(fr29), r29, five);
%pot(fr30) = setpot(pot(fr30), r30, two);
%pot(fr31) = setpot(pot(fr31), r31, three);
pot(fr32) = setpot(pot(fr32), r32, six);
pot(fr33) = setpot(pot(fr33), r33, five);
pot(fr34) = setpot(pot(fr34), r34, one);
%pot(fr35) = setpot(pot(fr35), r35, five);
%pot(fr36) = setpot(pot(fr36), r36, one);
fprintf('Sudoku starting conditions set up...\n');

% Setup the cluster graph %
% the sepset between C1 and C17 is v1 %
% cluster 1 has potential fr1, and is connected to cluster 37 via sepset v1 %

% clusters assignment for all cells involving horizontal and vertical rules %
fr = 19;
for c = 1:36
   for p = 1:18
      if ismember(c, pot(p).variables)
         cluster(c).pot = pot(fr);
         cluster(c).sepset{p + 36} = c;
      end;
   end;
   fr = fr + 1;
end;

fprintf('Cluster assignment for horizontal and vertical factors set up...\n');

% cluster assignment for all cells involving brick rules %
for p = 1:18
   cluster(p + 36).pot = pot(p);
end;
fprintf('Cluster assignment for brick factors set up...\n');

cluster = inferSepsets(cluster); % a sepset only has to be defined once, then it is added automatically by this function
fprintf('All sepsets are consistent...\n');

% Koller Slide 8.1.9: Intialise all messages to 1
cluster = initMessages(cluster, 6); % cluster(i).message(j) is the message to this cluster from cluster j
fprintf('All messages initialised to one...\n');

cluster = calculateBeliefs(cluster); % initialise beliefs cluster(i).belief
fprintf('Cluster beliefs initialised...\n');

sepsetBeliefs = calculateSepsetBeliefs(cluster); % calculate sepset beliefs
fprintf('Calculate sepset beliefs...\n');

% these messages do not change, so we can pass them in only once %
for i = 1:length(evidence_cells)
    for p = 1:18
        if ismember(evidence_cells(i), pot(p).variables)
            cluster = passMessage(cluster, evidence_cells(i), p + 36);
        end;
    end;
end;
fprintf('Pass evidence messages only once...\n');

sumSepsetChanges = [];
for k = 1:MAX_ITER
    % Store previous clusters and sepset beliefs %
    prevCluster = cluster;
    prevSepsetBeliefs = sepsetBeliefs;

    % Koller Slide 8.1.9: Select edge (i, j) and pass message %
    for i = 1:54
        for p = 1:18
          if ismember(i, pot(p).variables)
            if ismember(i, evidence_cells)
              cluster = passMessage(cluster, p + 36, i);
              %fprintf('%d -> %d\n', p + 36, i);
            else
              cluster = passMessage(cluster, i, p + 36);
              cluster = passMessage(cluster, p + 36, i);
              %fprintf('%d -> %d\n', i, p + 36);
              %fprintf('%d -> %d\n', p + 36, i);
            end;
          end;
       end;
    end;

    % Koller Slide 8.1.9: Compute beliefs
    cluster = calculateBeliefs(cluster);
    sepsetBeliefs = calculateSepsetBeliefs(cluster);

fprintf('\n\n');

% Decode by looking at max beliefs %
s = 0;
b = 0;
for i = v1:v36
    [tmp0, tmp1] = maxpot(cluster(i).belief, [], 0);
    s(i) = tmp1(1);
    b(i) = tmp0.table;
end;

clc;
fprintf('-------------------------------------------------------------------------\n');
fprintf('| %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) |\n', s(1), b(1), s(2), b(2), s(3), b(3), s(4), b(4), s(5), b(5), s(6), b(6));
fprintf('-------------------------------------------------------------------------\n');
fprintf('| %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) |\n', s(7), b(7), s(8), b(8), s(9), b(9), s(10), b(10), s(11), b(11), s(12), b(12));
fprintf('-------------------------------------------------------------------------\n');
fprintf('| %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) |\n', s(13), b(13), s(14), b(14), s(15), b(15), s(16), b(16), s(17), b(17), s(18), b(18));
fprintf('-------------------------------------------------------------------------\n');
fprintf('| %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) |\n', s(19), b(19), s(20), b(20), s(21), b(21), s(22), b(22), s(23), b(23), s(24), b(24));
fprintf('-------------------------------------------------------------------------\n');
fprintf('| %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) |\n', s(25), b(25), s(26), b(26), s(27), b(27), s(28), b(28), s(29), b(29), s(30), b(30));
fprintf('-------------------------------------------------------------------------\n');
fprintf('| %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) | %d (%.3f) |\n', s(31), b(31), s(32), b(32), s(33), b(33), s(34), b(34), s(35), b(35), s(36), b(36));
fprintf('-------------------------------------------------------------------------\n');
fprintf('iteration: %d\n', k);
pause(0.0001);

% Plot sepset beliefs sum %
plot(1:length(sumSepsetChanges), sumSepsetChanges, '-*');
xlabel('Iterations');
ylabel('Sum(KL distance) in sepset beliefs');
grid on;

    % check for distance convergence %
    change = sumSepsetKLChange(prevSepsetBeliefs, sepsetBeliefs);
    sumSepsetChanges = [sumSepsetChanges, change];
    if change <= SEPSET_SUM_CONVERGENCE_THRES
        fprintf('\n converged (%d iterations)\n', k)
        break
    end

end;
