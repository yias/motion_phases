



div=length(-0.150:0.050:2.50);
nUn=120;
historyTW='8';

% classesSet=[2,3,4,5,6];
classesSets=struct([]);
classesSets{1}=[2,3,4,5,6];
classesSets{2}=[2,3,5,6];
classesSets{3}=[2,5,6];
classesNames={'Rest','Precision\newlinedisk','Tripod','Thumb-4\newlinefingers','Pinch','Lateral'};
clrs={'b','g','r','m','y'};


%% for amputees

% load('data/subjectsDataTimimgs.mat')
% 
% muscleSets=struct([]);
% muscleSets{1}{1}=[2:11];
% muscleSets{1}{2}=[8:11];
% muscleSets{2}{1}=[1:6,8:12];
% muscleSets{2}{2}=[8:12];
% muscleSets{3}{1}=[1,4:12];
% muscleSets{3}{2}=[8:12];
% muscleSets{4}{1}=[1:12];
% muscleSets{4}{2}=[8:12];
% 
% 
% trialsForExclusion=struct([]);
% 
% % for amputee subject 1
% trialsForExclusion{1}{1}=[];
% trialsForExclusion{1}{2}=[];
% trialsForExclusion{1}{3}=[];
% trialsForExclusion{1}{4}=[];
% trialsForExclusion{1}{5}=[];
% trialsForExclusion{1}{6}=[];
% 
% % for amputee subject 2
% trialsForExclusion{2}{1}=[];
% trialsForExclusion{2}{2}=[];
% trialsForExclusion{2}{3}=[];
% trialsForExclusion{2}{4}=[];
% trialsForExclusion{2}{5}=[];
% trialsForExclusion{2}{6}=[];
% 
% % for amputee subject 3
% trialsForExclusion{3}{1}=[];
% trialsForExclusion{3}{2}=2;
% trialsForExclusion{3}{3}=2;
% trialsForExclusion{3}{4}=[];
% trialsForExclusion{3}{5}=[];
% trialsForExclusion{3}{6}=[];
% 
% % for amputee subject 4
% trialsForExclusion{4}{1}=[];
% trialsForExclusion{4}{2}=[];
% trialsForExclusion{4}{3}=[];
% trialsForExclusion{4}{4}=[26,28];
% trialsForExclusion{4}{5}=[];
% trialsForExclusion{4}{6}=[];

%% for able-bodied

load('data/ableBodiedData.mat')


muscleSets=struct([]);
muscleSets{1}{1}=[1:3,5,7:12];
muscleSets{1}{2}=8:12;
muscleSets{2}{1}=[1:10,12];
muscleSets{2}{2}=[8:10,12];
muscleSets{3}{1}=1:12;
muscleSets{3}{2}=8:12;
muscleSets{4}{1}=1:12;
muscleSets{4}{2}=8:12;
muscleSets{5}{1}=1:12;
muscleSets{5}{2}=8:12;



trialsForExclusion=struct([]);

% for able-bodied subject 1
trialsForExclusion{1}{1}=[];
trialsForExclusion{1}{2}=[];
trialsForExclusion{1}{3}=[11,21];
trialsForExclusion{1}{4}=[];
trialsForExclusion{1}{5}=[];
trialsForExclusion{1}{6}=[];

% for able-bodied subject 2
trialsForExclusion{2}{1}=[];
trialsForExclusion{2}{2}=29;
trialsForExclusion{2}{3}=[];
trialsForExclusion{2}{4}=[];
trialsForExclusion{2}{5}=[];
trialsForExclusion{2}{6}=15;

% for able-bodied subject 3
trialsForExclusion{3}{1}=[];
trialsForExclusion{3}{2}=[];
trialsForExclusion{3}{3}=[];
trialsForExclusion{3}{4}=[];
trialsForExclusion{3}{5}=[];
trialsForExclusion{3}{6}=[];

% for able-bodied subject 4
trialsForExclusion{4}{1}=[];
trialsForExclusion{4}{2}=[];
trialsForExclusion{4}{3}=[];
trialsForExclusion{4}{4}=[26,28];
trialsForExclusion{4}{5}=[];
trialsForExclusion{4}{6}=[];


% for able-bodied subject 5
trialsForExclusion{5}{1}=[];
trialsForExclusion{5}{2}=[];
trialsForExclusion{5}{3}=[];
trialsForExclusion{5}{4}=[];
trialsForExclusion{5}{5}=1;
trialsForExclusion{5}{6}=[];

%%

features={'RMS','WaveFormLength','SlopeChanges'};

ampSubjResults=struct([]);
%%

for sbj=5:5

    for mm=1:2
        for j=1:3
            
%             if sbj==3
%                 div=51;
%             else
%                 div=54;
%             end
            

            % create dataset
            %Fold1=createDataSets(sbjData{sbj},classesSets{j},div,'MuscleSet',muscleSets{sbj}{mm},'Normalization',true);%,'FeatureExtraction',features);
            [d1,d2,d3,testingSet,Fd1,Fd2,Fd3,FtestingSet]=createDataSets_byAngle(sbjData{sbj},classesSets{j},div,sbj,'MuscleSet',muscleSets{sbj}{mm},'Normalization',true,'FeatureExtraction',features);

            % classification
%             [ampSubjResults{sbj}{mm}{j}.performance,ampSubjResults{sbj}{mm}{j}.ConMat,ampSubjResults{sbj}{mm}{j}.oneESNperformance,ampSubjResults{sbj}{mm}{j}.oneESNConMat]=ESNCrossValidation(Fold1,div,nUn,historyTW,classesSets{j},classesNames(classesSets{j}),sbj,clrs{sbj},2.5,length(muscleSets{sbj}{mm}),length(classesSets{j}));
            
            % classification with ESN - dynamic classifier

%             [ampSubjResults{sbj}{mm}{j}.oneESNperformance,ampSubjResults{sbj}{mm}{j}.oneESNConMat]=ESNCrossValidation_byAngle(d1,d2,d3,testingSet,div,historyTW,classesSets{j},classesNames(classesSets{j}),sbj,clrs{sbj},2.5,length(muscleSets{sbj}{mm}),length(classesSets{j}));
        
            % classification with ESN - both approaches
            
            [ampSubjResults{sbj}{mm}{j}.oneESNperformance,ampSubjResults{sbj}{mm}{j}.oneESNConMat]=trainOneESNClassifier(d1,d2,d3,testingSet,div,historyTW,classesSets{j},classesNames(classesSets{j}),sbj,clrs{sbj},2.5,length(muscleSets{sbj}{mm}),length(classesSets{j}));
            
            
            % classification with linear SVM - dynamic classifier
            
            [ampSubjResults{sbj}{mm}{j}.oneESNperformance,ampSubjResults{sbj}{mm}{j}.oneESNConMat]=SVMCrossValidation_byAngle(Fd1,Fd2,Fd3,FtestingSet,div,historyTW,classesSets{j},classesNames(classesSets{j}),sbj,clrs{sbj},2.5,length(muscleSets{sbj}{mm}),length(classesSets{j}));
            
            % classification with linear SVM - one classifier

            [ampSubjResults{sbj}{mm}{j}.oneESNperformance,ampSubjResults{sbj}{mm}{j}.oneESNConMat]=trainOneLinearSVM(Fd1,Fd2,Fd3,FtestingSet,div,historyTW,classesSets{j},classesNames(classesSets{j}),sbj,clrs{sbj},2.5,length(muscleSets{sbj}{mm}),length(classesSets{j}));
            
            % classification with RBF SVM - dynamic classifier
            
            [ampSubjResults{sbj}{mm}{j}.oneESNperformance,ampSubjResults{sbj}{mm}{j}.oneESNConMat]=rbf_SVMClassification_byAngle(Fd1,Fd2,Fd3,FtestingSet,div,historyTW,classesSets{j},classesNames(classesSets{j}),sbj,clrs{sbj},2.5,length(muscleSets{sbj}{mm}),length(classesSets{j}));
            
            % classification with RBF SVM - one classifier
            
            [ampSubjResults{sbj}{mm}{j}.oneESNperformance,ampSubjResults{sbj}{mm}{j}.oneESNConMat]=trainOneRBFSVM(Fd1,Fd2,Fd3,FtestingSet,div,historyTW,classesSets{j},classesNames(classesSets{j}),sbj,clrs{sbj},2.5,length(muscleSets{sbj}{mm}),length(classesSets{j}));
            
        
            % classification with LDA 
            
            [ampSubjResults{sbj}{mm}{j}.oneESNperformance,ampSubjResults{sbj}{mm}{j}.oneESNConMat]=train_LDAClassifier(Fd1,Fd2,Fd3,FtestingSet,div,historyTW,classesSets{j},classesNames(classesSets{j}),sbj,clrs{sbj},2.5,length(muscleSets{sbj}{mm}),length(classesSets{j}));
        
        
        end
    end
end


%%
% 
% for sbj=1:4
% 
%     for mm=1:2
%         for j=1:3
%             
%             if sbj==3
%                 div=51;
%             else
%                 div=54;
%             end
% 
%             % create dataset
%             [Fold1,Fold2]=createDataSets(sbjData{sbj},classesSets{j},div,'MuscleSet',muscleSets{sbj}{mm},'Normalization',true,'FeatureExtraction',features);%,'FeatureExtraction',features);
% 
%             % classification with SVM
%             [AA,BB,SaMid,SaEnd,SbMid,SbEnd]=SVMCrossValidation(Fold1,div,historyTW,classesSets{j},classesNames(classesSets{j}),sbj,clrs{sbj},2.5,length(muscleSets{sbj}{mm}),length(classesSets{j}));
%             
%             % classification LDA
% %             [LDAResults]=LDAClassification(Fold1,div,historyTW,classesSets{j},classesNames(classesSets{j}),sbj,clrs{sbj},2.5,length(muscleSets{sbj}{mm}),length(classesSets{j}));
% %             figure(1)
% %             plot(-0.150:0.050:2.50,RRLDA.manyModels.testingScores.TestMVresults,'b')
% %             hold on
% %             plot(-0.150:0.050:2.50,RRLDA.manyModels.testingScores.addTestMVresults,'b--')
% %             plot(-0.150:0.050:2.50,RRLDA.oneModel.testingScores.TestMVresults,'r')
% %             plot(-0.150:0.050:2.50,RRLDA.oneModel.testingScores.addTestMVresults,'r--')
%         end
%     end
% end

