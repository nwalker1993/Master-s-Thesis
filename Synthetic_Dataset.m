%%last update: 16.09.19
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Part of the Master Thesis: Transformation of Spatial Contact Information
%%among limbs by Nicole Walker, 2019.
%%This script generates a synthetic dataset for leg mappings that include
%%all points that should be possibly reached by the respective legs. For
%%each point in space the inverse kinematics is calculated and then checked
%%if the resulting angles are within the range that was extracted from real
%%data angle distribution. If points are not reachable the point closest to
%%the target in Euclidean distance is used.
%%%%%%%%%%%%%%%%%%%%
%%edit Nicole Walker
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Requirements: Overlap_Targets_and_JointAngles_05_tar.mat
%              vec_InvKin.m
%              ang_tar.m
%              angle_dist.mat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%Transformation shift between the coordinate systems%%%%%%%%%%%%%%
%distance in x coordinates between the legs: +12 from middle to hind leg, +18
%from front to middle leg, so +30 from front to hind leg
%distance in y coordinates: -1 L1L2, +1 R1R2, +2 L1R1, +4 L2R2, +4 L3R3 
%the translation group acts transitively, therefore: e.g. L1R2
%+2y(from L1R1)+1y(from R1R2)+18x(from R1R2) = +3y +18x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Setup variables and everything

name = 'Overlap_Targets_and_JointAngles_05_tar.mat';
load(name);
%clear ANGLES_ variables to avoid cross talk of new ANGLES_ variables
clear ANGLES_*;
%Transformation shifts
%Front legs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shift_L1L2(1) = -TARGET_L1L2(1,4) + TARGET_L1L2(1,1);
shift_L1L2(2) = -TARGET_L1L2(1,5) + TARGET_L1L2(1,2);
shift_L1L2(3) = -TARGET_L1L2(1,6) + TARGET_L1L2(1,3);

shift_R1R2(1) = -TARGET_R1R2(1,4) + TARGET_R1R2(1,1);
shift_R1R2(2) = -TARGET_R1R2(1,5) + TARGET_R1R2(1,2);
shift_R1R2(3) = -TARGET_R1R2(1,6) + TARGET_R1R2(1,3);

shift_L1R1(1) = -TARGET_L1R1(1,4) + TARGET_L1R1(1,1);
shift_L1R1(2) = -TARGET_L1R1(1,5) + TARGET_L1R1(1,2);
shift_L1R1(3) = -TARGET_L1R1(1,6) + TARGET_L1R1(1,3);

%middle legs%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shift_L2L3(1) = -TARGET_L2L3(1,4) + TARGET_L2L3(1,1);
shift_L2L3(2) = -TARGET_L2L3(1,5) + TARGET_L2L3(1,2);
shift_L2L3(3) = -TARGET_L2L3(1,6) + TARGET_L2L3(1,3);

shift_R2R3(1) = -TARGET_R2R3(1,4) + TARGET_R2R3(1,1);
shift_R2R3(2) = -TARGET_R2R3(1,5) + TARGET_R2R3(1,2);
shift_R2R3(3) = -TARGET_R2R3(1,6) + TARGET_R2R3(1,3);

shift_L2R2(1) = -TARGET_L2R2(1,4) + TARGET_L2R2(1,1);
shift_L2R2(2) = -TARGET_L2R2(1,5) + TARGET_L2R2(1,2);
shift_L2R2(3) = -TARGET_L2R2(1,6) + TARGET_L2R2(1,3);

%hind legs%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shift_L3R3(1) = -TARGET_L3R3(1,4) + TARGET_L3R3(1,1);
shift_L3R3(2) = -TARGET_L3R3(1,5) + TARGET_L3R3(1,2);
shift_L3R3(3) = -TARGET_L3R3(1,6) + TARGET_L3R3(1,3);

%create a struct with shift information for each sender leg to all others
shift.L1 = [];
shift.L2 = [];
shift.L3 = [];
shift.R1 = [];
shift.R2 = [];
shift.R3 = [];

%%%%%%%%%%%%%%%%Left legs%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shift.L1(:,1) = -shift_L1L2';
shift.L1(:,2) = -(shift_L1L2' + shift_L2L3');
shift.L1(:,3) = -shift_L1R1';
shift.L1(:,4) = -(shift_L1R1' + [-18 -1 0]');
shift.L1(:,5) = -(shift_L1R1'+ [-18 -1 0]' + shift_R2R3');

shift.L2(:,1) = shift_L1L2';
shift.L2(:,2) = -shift_L2L3';
shift.L2(:,3) = -(shift_L2R2' + [18 1 0]');
shift.L2(:,4) = -shift_L2R2';
shift.L2(:,5) = -(shift_L2L3' + shift_L3R3');

shift.L3(:,1) = -(-shift_L2L3' - shift_L1L2');
shift.L3(:,2) = shift_L2L3';
shift.L3(:,3) = -(shift_L3R3' - shift_R2R3' - [-18 -1 0]');
shift.L3(:,4) = -(shift_L3R3' - shift_R2R3');
shift.L3(:,5) = -shift_L3R3';

%%%%%%%%%%%%%%%%%Right legs%%%%%%%%%%%%%%%%%%%%%%%%%%%
shift.R1(:,1) = shift_L1R1';
shift.R1(:,2) = shift_L1R1' - [-18 1 0]';
shift.R1(:,3) = shift_L1R1' - [-18 1 0]' - shift_L2L3';
shift.R1(:,4) = -shift_R1R2';
shift.R1(:,5) = -(shift_R1R2' + shift_R2R3');

shift.R2(:,1) = shift_R1R2' + shift_L1R1';
shift.R2(:,2) = shift_L2R2';
shift.R2(:,3) = shift_L2R2' - shift_L2L3';
shift.R2(:,4) = shift_R1R2';
shift.R2(:,5) = -shift_R2R3';

shift.R3(:,1) = -(-shift_R2R3' - shift_R1R2' - shift_L1R1');
shift.R3(:,2) = shift_L3R3' + shift_L2L3';
shift.R3(:,3) = shift_L3R3';
shift.R3(:,4) = -(-shift_R2R3' - shift_R1R2');
shift.R3(:,5) = shift_R2R3';


%% First create 1mm equidistant points in space to sample from
%find the extreme x,y,z coordinates that are used in the dataset
all_targets = [TARGET_L1L2; TARGET_L1R1; TARGET_L2L3; ...
    TARGET_L2R2; TARGET_L3R3; TARGET_R1R2; TARGET_R2R3];
lev = max(all_targets(:,3))+15;
depr = min(all_targets(:,3))-15;
ext = max([all_targets(:,2); all_targets(:,4)])+18;
protr = max(all_targets(:,1))+18;
retra = min(all_targets(:,1))-18;
x = retra:1:protr;
y = -ext:1:ext;
z = depr:1:lev;
[X,Y,Z] = meshgrid(x,y,z);
sample_targets = [X(:), Y(:), Z(:)];
% %plot to check
% scatter3(X(:),Y(:),Z(:), 'MarkerFaceAlpha',0.05, 'MarkerEdgeAlpha', 0.05)
% % hold on
% % line([-20, 20], [0 0], [0 0])
clearvars -except name sample_targets mn shift
%% Calculate the inverse kinematics for each point and leg
legnames = {'L1', 'L2', 'L3', 'R1', 'R2', 'R3'};
ANGLES = cell(1,6);
TARGET = cell(1,6);
for leg = 1:length(legnames)
[ANGLES{leg}, TARGET{leg}] = ang_tar(mn, legnames(leg), sample_targets);
end

%% get the labels if a target is reachable or not, i.e. shift the sender leg 
%coordinates accordingly and see if ismember
ANGLES_yn = ANGLES;
TARGET_yn = TARGET;
TARGET_map = TARGET;
ANGLES_map = ANGLES;

load('angle_dist.mat')
ranges = cell(1,6);
ranges{1} = ex_front;
ranges{2} = ex_front;
ranges{3} = ex_mid;
ranges{4} = ex_mid;
ranges{5} = ex_hind;
ranges{6} = ex_hind;

order = [2, 3, 4, 5, 6;
        1, 3, 4, 5, 6;
        1, 2, 4, 5, 6;
        1, 2, 3, 5, 6;
        1, 2, 3, 4, 6;
        1, 2, 3, 4, 5];

fields = fieldnames(shift);
%loops through the legs and adds 0/1 compared to all other legs if the
%target is reachable, which results in a x8 matrix (1:3 targets 3:8 position
%reachable for receiver legs)
for leg = 1:numel(fields)
    for rec = 1:5
        new_tar(:,1) = TARGET{leg}(:,1) + shift.(fields{leg})(1,rec);
        new_tar(:,2) = TARGET{leg}(:,2) + shift.(fields{leg})(2,rec);
        new_tar(:,3) = TARGET{leg}(:,3) + shift.(fields{leg})(3,rec);
        
        TARGET_yn{leg} = [TARGET_yn{leg}, ismember(new_tar, TARGET{order(leg,rec)}(:,1:3), 'rows')];
        ANGLES_yn{leg} = [ANGLES_yn{leg}, ismember(new_tar, TARGET{order(leg,rec)}(:,1:3), 'rows')];
        
        % as a last step a true mapping dataset is needed. As each receiver leg
        %won't have the same amount of reachable points to the sender leg, each leg
        %output would have a different size. This is problematic to use for the
        %neural network. Therefore the 0/1 information is used to fill in the
        %existing mapping whenever there is 1 and put random angles within the
        %range whenever the entry is 0 (= noise that will not be learned)
        non_reachable = find(TARGET_yn{leg}(:,rec+3) == 0);
        %a = ranges{order(leg, rec)};
        %r1 = (a(2,1)-a(1,1)).*rand(length(non_reachable),1) + a(1,1);
        %r2 = (a(2,2)-a(1,2)).*rand(length(non_reachable),1) + a(1,2);
        %r3 = (a(2,3)-a(1,3)).*rand(length(non_reachable),1) + a(1,3);
        
        %although it is probably better to just pick the point that has
        %minimum Euclidean distance to the nearest reachable point
        
        reachable = find(TARGET_yn{leg}(:,rec+3) == 1);
        %finds the indices of nearest points of non-reachable targets in 
        %the reachable ones
        [k,d] = dsearchn(TARGET_yn{leg}(reachable,1:3),...
                         TARGET_yn{leg}(non_reachable,1:3)); 
        
        %now we onnly have the nearest points in sender leg coordinates.
        %Therefore we transform to receiver coodinates
        a = [];
        a(:,1) = TARGET_yn{leg}(reachable(k),1) + shift.(fields{leg})(1,rec);
        a(:,2) = TARGET_yn{leg}(reachable(k),2) + shift.(fields{leg})(2,rec);
        a(:,3) = TARGET_yn{leg}(reachable(k),3) + shift.(fields{leg})(3,rec);
        
        %now we need to extract the indices of the receiver targets, so we
        %can extract the corresponding angles and fill them in the data set
        y=zeros(length(a),1);
        for nearest = 1:length(a)
            y(nearest) = find(ismember(TARGET_yn{order(leg,rec)}(:,1:3),...
                a(nearest,1:3), 'rows'));
        end
                     
        %use the indices to find the angles/targets to these positions
        r1 = ANGLES_yn{order(leg,rec)}(y,1);
        r2 = ANGLES_yn{order(leg,rec)}(y,2);
        r3 = ANGLES_yn{order(leg,rec)}(y,3);
        s1 = TARGET_yn{order(leg,rec)}(y,1);
        s2 = TARGET_yn{order(leg,rec)}(y,2);
        s3 = TARGET_yn{order(leg,rec)}(y,3);
        
        x = new_tar(TARGET_yn{leg}(:,rec+3) == 1, 1:3);
        x2 = new_tar(TARGET_yn{leg}(:,rec+3) == 1, 1:3);
        
        new_ang = zeros(length(ANGLES_map{leg}),3);
        new_target = zeros(length(TARGET_map{leg}),3);
        
        new_ang(TARGET_yn{leg}(:,rec+3) == 1, 1:3) =  ANGLES{order(leg,rec)}...
            (ismember(TARGET{order(leg,rec)}(:,1:3), x(:,1:3), 'rows'), 1:3);
        new_target =  new_tar;
        new_ang(TARGET_yn{leg}(:,rec+3) == 0, 1:3) = [r1,r2,r3];
        new_target(TARGET_yn{leg}(:,rec+3) == 0, 1:3) = [s1,s2,s3];
        
        ANGLES_map{leg} = [ANGLES_map{leg}, new_ang];
        TARGET_map{leg} = [TARGET_map{leg}, new_target];
    end
    new_tar = [];
end
%% check your Target points
% figure
% scatter3(TARGET_map_clean{5}(:,1), TARGET_map_clean{5}(:,2), TARGET_map_clean{5}(:,3), 'MarkerFaceAlpha',0.05, 'MarkerEdgeAlpha', 0.05);
% hold on
% scatter3(-26,0, -32)
% hold on
% scatter3(-16,0, -23)

scatter3(TARGET_map_clean{4}(:,1), TARGET_map_clean{4}(:,2), TARGET_map_clean{4}(:,3), 'MarkerFaceAlpha',0.05, 'MarkerEdgeAlpha', 0.05);
xlabel('x')
ylabel('y')
% hold on
% scatter3(-5, -6, -27)
% hold on
% scatter3(-8, -6, -19)
all=[];
for x = 1:10847
 all = [all; TARGET_map_clean{6}(x,4:6)-shift.R3(:,1)';...
     TARGET_map_clean{6}(x,7:9)-shift.R3(:,2)';...
     TARGET_map_clean{6}(x,10:12)-shift.R3(:,3)';...
     TARGET_map_clean{6}(x,13:15)-shift.R3(:,4)';...
     TARGET_map_clean{6}(x,16:18)-shift.R3(:,5)'];
end

mean(all)
scatter(all(:,1), all(:,2))
hold on 
quiver(0,0,mid(1), mid(2))
xlabel('x')
ylabel('y')
%% we don't want all possibly reachable points as there are probably many 
% that have no overlap with any of the other legs. The following picks only
% sender points if there is a "1" at any of the receiver legs
ANGLES_yn_clean = cell(1,6);
ANGLES_map_clean = cell(1,6);
ANGLES_map_yn = cell(1,6);
TARGET_map_clean = cell(1,6);
TARGET_map_yn = cell(1,6);

for leg = 1:6
    row = 1;
    for samples = 1:length(ANGLES_yn{leg})
        if sum(ANGLES_yn{leg}(samples,4:8)) > 0
            ANGLES_yn_clean{leg}(row,:) = ANGLES_yn{leg}(samples,:);
            ANGLES_map_clean{leg}(row,:) = ANGLES_map{leg}(samples,:);
            TARGET_map_clean{leg}(row,:) = TARGET_map{leg}(samples,:);
            row = row+1;
        end
    end
    %here both classification (0/1) and regression mappings are concluded
    %in one dataset (4 outputs per leg)
    ANGLES_map_yn{leg} = [ANGLES_map_clean{leg}(:,1:6), ANGLES_yn_clean{leg}(:,4),...
        ANGLES_map_clean{leg}(:,7:9), ANGLES_yn_clean{leg}(:,5), ...
        ANGLES_map_clean{leg}(:,10:12), ANGLES_yn_clean{leg}(:,6),...
        ANGLES_map_clean{leg}(:,13:15), ANGLES_yn_clean{leg}(:,7),...
        ANGLES_map_clean{leg}(:,16:18), ANGLES_yn_clean{leg}(:,8),];
    
     TARGET_map_yn{leg} = [TARGET_map_clean{leg}(:,1:6), ANGLES_yn_clean{leg}(:,4),...
        TARGET_map_clean{leg}(:,7:9), ANGLES_yn_clean{leg}(:,5), ...
        TARGET_map_clean{leg}(:,10:12), ANGLES_yn_clean{leg}(:,6),...
        TARGET_map_clean{leg}(:,13:15), ANGLES_yn_clean{leg}(:,7),...
        TARGET_map_clean{leg}(:,16:18), ANGLES_yn_clean{leg}(:,8),];
end
%% plot the volumes
% figure
% scatter3(TARGET_map_clean{1}(:,1), TARGET_map_clean{1}(:,2), TARGET_map_clean{1}(:,3), 'MarkerFaceAlpha',0.05, 'MarkerEdgeAlpha', 0.05);
% hold on
% scatter3(31.347156278653340-31.347156278653340, 1.325851850865031, -0.728640161959598)
% hold on
% scatter3(12.899415525121622-31.347156278653340, 1.696763682818173, -1.108390444810411)
% hold on
% scatter3(1.344419992681526-31.347156278653340, 1.667047922977852, -1.053028278241662)
% hold on
% scatter3(31.347156278653340-31.347156278653340, -1.325851850865031, -0.728640161959598)
% hold on
% scatter3(12.899415525121622-31.347156278653340, -1.696763682818173, -1.108390444810411)
% hold on
% scatter3(1.344419992681526-31.347156278653340, -1.667047922977852, -1.053028278241662)
% hold on 
% scatter3(TARGET_map_clean{2}(:,1)+shift.L2(1,4), TARGET_map_clean{2}(:,2)+shift.L2(2,4),...
%     TARGET_map_clean{2}(:,3), 'MarkerFaceAlpha',0.05, 'MarkerEdgeAlpha', 0.05);
% hold on 
% scatter3(TARGET_map_clean{3}(:,1)+shift.L3(1,5), TARGET_map_clean{3}(:,2)+shift.L3(2,5),...
%     TARGET_map_clean{3}(:,3), 'MarkerFaceAlpha',0.05, 'MarkerEdgeAlpha', 0.05);
% hold on 
% scatter3(TARGET_map_clean{4}(:,1), TARGET_map_clean{4}(:,2),...
%     TARGET_map_clean{4}(:,3), 'MarkerFaceAlpha',0.05, 'MarkerEdgeAlpha', 0.05);
% hold on 
% scatter3(TARGET_map_clean{5}(:,1), TARGET_map_clean{5}(:,2),...
%     TARGET_map_clean{5}(:,3), 'MarkerFaceAlpha',0.05, 'MarkerEdgeAlpha', 0.05);
% hold on 
% scatter3(TARGET_map_yn{6}(:,1), TARGET_map_yn{6}(:,2),...
%     TARGET_map_yn{6}(:,3), 'MarkerFaceAlpha',0.05, 'MarkerEdgeAlpha', 0.05);
% hold on 
% scatter3(TARGET_map_yn{6}(:,12)-shift.R3(1,3), TARGET_map_yn{6}(:,13)-shift.R3(2,3),...
%     TARGET_map_yn{6}(:,14), 'MarkerFaceAlpha',0.05, 'MarkerEdgeAlpha', 0.05);

%% save
save 'synth_dataset.mat' ANGLES*  TARGET* 'shift' 'mn'
