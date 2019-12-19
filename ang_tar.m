function [ANGLES, TARGET] = ang_tar(mn, leg, sample_targets)
%%last update: 26.06.19
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Part of the Master Thesis: Transformation of Spatial Contact Information
%%among limbs by Nicole Walker, 2019.
%this function goes through all provided example target point and
%calculates the angles. Then it excludes all the points that are not within
%the specific angle ranges as well as NaN and complex results and in the
%end provides you with the target points that are within reach and the
%respective angles.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%edit Nicole Walker
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Requirements: mn - leg lengths and psi/phi for inverse kinematics
%              leg - leg name as a string (e.g. 'R1', 'R2', etc.)
%              sample_targets - the target points that should be tested
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%defines the extreme joint angles values of the 3 joints
DOF3_R1 = [-53.5684,  -67.0934,   42.1081;
    89.9688 ,  69.0094 , 161.1893];

DOF3_L1 = [-80.0180,  -75.8842,   38.6104;
    108.2690,   68.2173,  160.1743];

DOF3_R2 = [-49.5907,  -33.8995,   32.7447;
    56.6350,   62.8382,  153.2583];

DOF3_L2 = [-46.0489,  -33.8054,   36.9627;
    53.5319,   57.2767,  155.6095];

DOF3_R3 = [-80.3798,  -40.8297,   38.1808;
    31.5833,   58.9604,  160.5845];

DOF3_L3 = [-79.9838,  -36.2205,   38.6199;
    25.6916,   62.3599,  165.4619];

%average the sides for symmetry
ex_front(1,:) = mean([DOF3_L1(1,:);DOF3_R1(1,:)]);
ex_front(2,:) = mean([DOF3_L1(2,:);DOF3_R1(2,:)]);

ex_mid(1,:) = mean([DOF3_L2(1,:);DOF3_R2(1,:)]);
ex_mid(2,:) = mean([DOF3_L2(2,:);DOF3_R2(2,:)]);

ex_hind(1,:) = mean([DOF3_L3(1,:);DOF3_R3(1,:)]);
ex_hind(2,:) = mean([DOF3_L3(2,:);DOF3_R3(2,:)]);

%calculate the angles and exclude points that are not within the angle
%range defined above

if ~isempty(strfind(leg{1}, '1')) == true
    
    if strcmp(leg, 'R1') == 1
        leg_angles = vec_InvKin(sample_targets, [mn.R1.cox.LENGTH, mn.R1.fem.LENGTH,...
            mn.R1.tib.LENGTH+mn.R1.tar.LENGTH] , mn.R1.phi, mn.R1.psi);
        
        indx = all(leg_angles(:,1) < -ex_front(1,1)  & leg_angles(:,1) > -ex_front(2,1), 2);
        indx = find(indx == 1);
        leg_angles = leg_angles(indx,:);
        leg_targets = sample_targets(indx,:);
        
    else
        leg_angles = vec_InvKin(sample_targets, [mn.L1.cox.LENGTH, mn.L1.fem.LENGTH,...
            mn.L1.tib.LENGTH+mn.L1.tar.LENGTH] , mn.L1.phi, mn.L1.psi);
        
        indx = all(leg_angles(:,1) > ex_front(1,1)  & leg_angles(:,1) < ex_front(2,1), 2);
        indx = find(indx == 1);
        leg_angles = leg_angles(indx,:);
        leg_targets = sample_targets(indx,:);
    end
    
    indx =  all(leg_angles(:,2) > ex_front(1,2) & leg_angles(:,2) < ex_front(2,2),2);
    indx = find(indx == 1);
    leg_angles = leg_angles(indx,:);
    leg_targets = leg_targets(indx,:);
    
    indx = all(leg_angles(:,3) > ex_front(1,3) & leg_angles(:,3) < ex_front(2,3),2);
    indx = find(indx == 1);
    leg_angles = leg_angles(indx,:);
    leg_targets = leg_targets(indx,:);
    
    
elseif ~isempty(strfind(leg{1}, '2')) == true
    
    if strcmp(leg, 'R2') == 1
        leg_angles = vec_InvKin(sample_targets, [mn.R2.cox.LENGTH, mn.R2.fem.LENGTH,...
            mn.R2.tib.LENGTH+mn.R2.tar.LENGTH] , mn.R2.phi, mn.R2.psi);
        
        indx = all(leg_angles(:,1) < -ex_mid(1,1)  & leg_angles(:,1) > -ex_mid(2,1), 2);
        indx = find(indx == 1);
        leg_angles = leg_angles(indx,:);
        leg_targets = sample_targets(indx,:);
        
    else
        leg_angles = vec_InvKin(sample_targets, [mn.L2.cox.LENGTH, mn.L2.fem.LENGTH,...
            mn.L2.tib.LENGTH+mn.L2.tar.LENGTH] , mn.L2.phi, mn.L2.psi);
        
        indx = all(leg_angles(:,1) > ex_mid(1,1)  & leg_angles(:,1) < ex_mid(2,1), 2);
        indx = find(indx == 1);
        leg_angles = leg_angles(indx,:);
        leg_targets = sample_targets(indx,:);
    end
    
    indx =  all(leg_angles(:,2) > ex_mid(1,2) & leg_angles(:,2) < ex_mid(2,2),2);
    indx = find(indx == 1);
    leg_angles = leg_angles(indx,:);
    leg_targets = leg_targets(indx,:);
    
    indx = all(leg_angles(:,3) > ex_mid(1,3) & leg_angles(:,3) < ex_mid(2,3),2);
    indx = find(indx == 1);
    leg_angles = leg_angles(indx,:);
    leg_targets = leg_targets(indx,:);
    
elseif ~isempty(strfind(leg{1}, '3')) == true
    
    if strcmp(leg, 'R3') == 1
        leg_angles = vec_InvKin(sample_targets, [mn.R3.cox.LENGTH, mn.R3.fem.LENGTH,...
            mn.R3.tib.LENGTH+mn.R3.tar.LENGTH] , mn.R3.phi, mn.R3.psi);
        
        indx = all(leg_angles(:,1) < -ex_hind(1,1)  & leg_angles(:,1) > -ex_hind(2,1), 2);
        indx = find(indx == 1);
        leg_angles = leg_angles(indx,:);
        leg_targets = sample_targets(indx,:);
        
    else
        leg_angles = vec_InvKin(sample_targets, [mn.L3.cox.LENGTH, mn.L3.fem.LENGTH,...
            mn.L3.tib.LENGTH+mn.L3.tar.LENGTH] , mn.L3.phi, mn.L3.psi);
        
        indx = all(leg_angles(:,1) > ex_hind(1,1)  & leg_angles(:,1) < ex_hind(2,1), 2);
        indx = find(indx == 1);
        leg_angles = leg_angles(indx,:);
        leg_targets = sample_targets(indx,:);
    end
    
    indx =  all(leg_angles(:,2) > ex_hind(1,2) & leg_angles(:,2) < ex_hind(2,2),2);
    indx = find(indx == 1);
    leg_angles = leg_angles(indx,:);
    leg_targets = leg_targets(indx,:);
    
    indx = all(leg_angles(:,3) > ex_hind(1,3) & leg_angles(:,3) < ex_hind(2,3),2);
    indx = find(indx == 1);
    leg_angles = leg_angles(indx,:);
    leg_targets = leg_targets(indx,:);
    
end
%remove NaN and complex numbers
leg_anglesmod = leg_angles(all(~isnan(leg_angles),2),:);
leg_targetsmod = leg_targets(all(~isnan(leg_targets),2),:);
ANGLES = leg_anglesmod(all(leg_anglesmod == real(leg_anglesmod),2),:);
TARGET = leg_targetsmod(all(leg_anglesmod == real(leg_anglesmod),2),:);

end