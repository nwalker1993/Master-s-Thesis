%last update 10.07.2019
%The new data is available and the contralateral overlaps for all leg pairs
%can be computed now.
%27.05.2019
%This script adds the tarsus length to the leg length and recomputes the
%inverse kinematics for the joint angles. It works either with 
%Overlap_Targets_and_JointAngles_04.mat or with the _05.mat version. The
%_05.mat version also produces the joint angles for contralateral overlaps
%but only for L1R1 so far as there is little to no data for the other
%pairs of contralateral legs. This might change after the "turn" data will
%be available.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%NOTE!! version _05.mat is now updated with all leg pairs. Also %%%%%%%%
%%%%it might be that the tarsus length was already added!!!!%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Requirements: Overlap_Targets_and_JointAngles_04.mat (or _05.mat)
%              vec_InvKin.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%edit: Nicole Walker

%first load Overlap_Targets_and_JointAngles_04.mat (for contralateral _05.mat) to save the TARGET_ variables
name = 'Overlap_Targets_and_JointAngles_04.mat';
load(name);
%clear ANGLES_ variables to avoid cross talk of new ANGLES_ variables
clear ANGLES_*;
%Add tarsus length to tibia and calculate new angles
ANGLES_L1L2 = horzcat(vec_InvKin(TARGET_L1L2(:, 1:3), [mn.L1.cox.LENGTH, mn.L1.fem.LENGTH, mn.L1.tib.LENGTH+mn.L1.tar.LENGTH] , mn.L1.phi, mn.L1.psi), ...
    vec_InvKin(TARGET_L1L2(:, 4:6), [mn.L2.cox.LENGTH, mn.L2.fem.LENGTH, mn.L2.tib.LENGTH+mn.L2.tar.LENGTH] , mn.L2.phi, mn.L2.psi));

%find all cases where argument of acos is out of range [-1,1] and produce complex numbers
[ANGLES_R1R2]=horzcat(vec_InvKin(TARGET_R1R2(:, 1:3), [mn.R1.cox.LENGTH, mn.R1.fem.LENGTH, mn.R1.tib.LENGTH+mn.R1.tar.LENGTH] , mn.R1.phi, mn.R1.psi), ...
    vec_InvKin(TARGET_R1R2(:, 4:6), [mn.R2.cox.LENGTH, mn.R2.fem.LENGTH, mn.R2.tib.LENGTH+mn.R2.tar.LENGTH] , mn.R2.phi, mn.R2.psi));

for i = 1:6
    for k = 1:size(ANGLES_R1R2,1)
       realabcR1R2(k, i) = isreal(ANGLES_R1R2(k, i)); 
    end
end

%complex_indicesR1R2 = find(realabcR1R2(:,2) == 0);
[complex_rowR1R2, complex_columnR1R2] = find(realabcR1R2 == 0);
%complex_positionR1R2 = TARGET_R1R2(complex_indicesR1R2, 4:6); %gives 3D coordinates of non-working cases for plotting
ANGLES_R1R2(complex_rowR1R2,complex_columnR1R2) = NaN;

%same for R2R3
[ANGLES_R2R3]=horzcat(vec_InvKin(TARGET_R2R3(:, 1:3), [mn.R2.cox.LENGTH, mn.R2.fem.LENGTH, mn.R2.tib.LENGTH+mn.R2.tar.LENGTH] , mn.R2.phi, mn.R2.psi), ...
    vec_InvKin(TARGET_R2R3(:, 4:6), [mn.R3.cox.LENGTH, mn.R3.fem.LENGTH, mn.R3.tib.LENGTH+mn.R3.tar.LENGTH] , mn.R3.phi, mn.R3.psi));

for i = 1:6
    for k = 1:size(ANGLES_R2R3,1)
       realabcR2R3(k, i) = isreal(ANGLES_R2R3(k, i)); 
    end
end

%complex_indicesR2R3 = find(realabcR2R3(:,2) == 0);
[complex_rowR2R3, complex_columnR2R3] = find(realabcR2R3 == 0);
%complex_positionR2R3 = TARGET_R2R3(complex_indicesR2R3, 4:6);
ANGLES_R2R3(complex_rowR2R3,complex_columnR2R3) = NaN;

%also for L2L3
[ANGLES_L2L3]= horzcat(vec_InvKin(TARGET_L2L3(:, 1:3), [mn.L2.cox.LENGTH, mn.L2.fem.LENGTH, mn.L2.tib.LENGTH+mn.L2.tar.LENGTH] , mn.L2.phi, mn.L2.psi), ...
    vec_InvKin(TARGET_L2L3(:, 4:6), [mn.L3.cox.LENGTH, mn.L3.fem.LENGTH, mn.L3.tib.LENGTH+mn.L3.tar.LENGTH] , mn.L3.phi, mn.L3.psi));

for i = 1:6
    for k = 1:size(ANGLES_L2L3,1)
       realabcL2L3(k, i) = isreal(ANGLES_L2L3(k, i)); 
    end
end

%complex_indicesR2R3 = find(realabcR2R3(:,2) == 0);
[complex_rowL2L3, complex_columnL2L3] = find(realabcL2L3 == 0);
%complex_positionR2R3 = TARGET_R2R3(complex_indicesR2R3, 4:6);
ANGLES_L2L3(complex_rowL2L3,complex_columnL2L3) = NaN;


if strcmp(name, 'Overlap_Targets_and_JointAngles_05.mat') == 1
    
    ANGLES_L2R2 = horzcat(vec_InvKin(TARGET_L2R2(:, 1:3), [mn.L2.cox.LENGTH, mn.L2.fem.LENGTH, mn.L2.tib.LENGTH+mn.L2.tar.LENGTH] , mn.L2.phi, mn.L2.psi), ...
        vec_InvKin(TARGET_L2R2(:, 4:6), [mn.R2.cox.LENGTH, mn.R2.fem.LENGTH, mn.R2.tib.LENGTH+mn.R2.tar.LENGTH] , mn.R2.phi, mn.R2.psi));
    ANGLES_L3R3 = horzcat(vec_InvKin(TARGET_L3R3(:, 1:3), [mn.L3.cox.LENGTH, mn.L3.fem.LENGTH, mn.L3.tib.LENGTH+mn.L3.tar.LENGTH] , mn.L3.phi, mn.L3.psi), ...
        vec_InvKin(TARGET_L3R3(:, 4:6), [mn.R3.cox.LENGTH, mn.R3.fem.LENGTH, mn.R3.tib.LENGTH+mn.R3.tar.LENGTH] , mn.R3.phi, mn.R3.psi));
    
    %and same for contralateral L1R1 which for some reason has this issue
    %for both legs
    [abcL1R1]=vec_InvKin(TARGET_L1R1(:, 4:6), [mn.R1.cox.LENGTH, mn.R1.fem.LENGTH, mn.R1.tib.LENGTH+mn.R1.tar.LENGTH] , mn.R1.phi, mn.R1.psi);
    [abcR1L1]=vec_InvKin(TARGET_L1R1(:, 1:3), [mn.L1.cox.LENGTH, mn.L1.fem.LENGTH, mn.L1.tib.LENGTH+mn.L1.tar.LENGTH] , mn.L1.phi, mn.L1.psi);
   
    for i = 1:3
        for k = 1:size(abcL1R1,1)
           realabcL1R1(k, i) = isreal(abcL1R1(k, i)); 
           realabcR1L1(k, i) = isreal(abcR1L1(k, i)); 
        end
    end

    %complex_indicesL1R1 = find(realabcL1R1(:,2) == 0);
    %complex_indicesR1L1 = find(realabcR1L1(:,2) == 0);
    [complex_rowL1R1, complex_columnL1R1] = find(realabcL1R1 == 0);
    [complex_rowR1L1, complex_columnR1L1] = find(realabcR1L1 == 0);
    
    %complex_positionL1R1 = TARGET_L1R1(complex_indicesL1R1, 4:6);
    %complex_positionR1L1 = TARGET_L1R1(complex_indicesR1L1, 1:3);
    abcL1R1(complex_rowL1R1,complex_columnL1R1) = NaN;
    abcR1L1(complex_rowR1L1,complex_columnR1L1) = NaN;

    ANGLES_L1R1 = horzcat(abcL1R1, abcR1L1);
end

if strcmp(name, 'Overlap_Targets_and_JointAngles_05.mat') == 1
    save([name(1:end-4),'_tar.mat'], 'ANGLES_L1L2', 'ANGLES_L2L3', 'ANGLES_R1R2', 'ANGLES_R2R3', 'ANGLES_L1R1', 'ANGLES_L2R2', 'ANGLES_L3R3', 'TARGET_L1L2', 'TARGET_L2L3', 'TARGET_R1R2', 'TARGET_R2R3', 'TARGET_L1R1', 'TARGET_L2R2', 'TARGET_L3R3')
else
    save([name(1:end-4),'_tar.mat'], 'ANGLES_L1L2', 'ANGLES_L2L3', 'ANGLES_R1R2', 'ANGLES_R2R3', 'TARGET_L1L2', 'TARGET_L2L3', 'TARGET_R1R2', 'TARGET_R2R3')
end
    
%%
% figure(1)
% subplot(2,1,1)
% plotv(complex_positionL1R1')
% subplot(2,1,2)
% plotv(TARGET_L1R1(1:300, 4:6)')
% 
% figure(2)
% subplot(2,1,1)
% plotv(complex_positionL1R1')
% subplot(2,1,2)
% plotv(TARGET_L1R1(1:300, 4:6)')
% 
% abslenR1R2 = vecnorm(TARGET_L1R1(complex_indicesR1L1)', 3)';
% abslenR2R3 = vecnorm(TARGET_L1R1(complex_indicesL1R1)', 3)';