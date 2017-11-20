%% for amputees

load('data/subjectsDataTimimgs.mat')

muscleSets=struct([]);
muscleSets{1}{1}=[2:11];
muscleSets{1}{2}=[8:11];
muscleSets{2}{1}=[1:6,8:12];
muscleSets{2}{2}=[8:12];
muscleSets{3}{1}=[1,4:12];
muscleSets{3}{2}=[8:12];
muscleSets{4}{1}=[1:12];
muscleSets{4}{2}=[8:12];


trialsForExclusion=struct([]);

% for amputee subject 1
trialsForExclusion{1}{1}=[];
trialsForExclusion{1}{2}=[];
trialsForExclusion{1}{3}=[];
trialsForExclusion{1}{4}=[];
trialsForExclusion{1}{5}=[];
trialsForExclusion{1}{6}=[];

% for amputee subject 2
trialsForExclusion{2}{1}=[];
trialsForExclusion{2}{2}=[];
trialsForExclusion{2}{3}=[];
trialsForExclusion{2}{4}=[];
trialsForExclusion{2}{5}=[];
trialsForExclusion{2}{6}=[];

% for amputee subject 3
trialsForExclusion{3}{1}=[];
trialsForExclusion{3}{2}=2;
trialsForExclusion{3}{3}=2;
trialsForExclusion{3}{4}=[];
trialsForExclusion{3}{5}=[];
trialsForExclusion{3}{6}=[];

% for amputee subject 4
trialsForExclusion{4}{1}=[];
trialsForExclusion{4}{2}=[];
trialsForExclusion{4}{3}=[];
trialsForExclusion{4}{4}=[26,28];
trialsForExclusion{4}{5}=[];
trialsForExclusion{4}{6}=[];


%%
% features={'RMS','WaveFormLength','SlopeChanges'};
% features={'RMS','WaveFormLength'};
features={'Average'};
%%

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

for sbj=1:1
    
    [d1,d2,d3,testingSet,Fd1,Fd2,Fd3,FtestingSet]=createDataSets_byAngle(sbjData{sbj},2:6,15,sbj,'MuscleSet',muscleSets{sbj}{1},'Normalization',true,'FeatureExtraction',features);
    
    FD1data=[];
    FD1labels=[];
    
    for i=1:length(Fd1)
        
        for j=1:length(Fd1{i}.data)
            FD1data=[FD1data;Fd1{i}.data{j}];
            
            for k=1:length(Fd1{i}.labels{j})
                FD1labels=[FD1labels;Fd1{i}.labels{j}{k}];
            end
        end
        
    end
    
    
    FD2data=[];
    FD2labels=[];
    
    for i=1:length(Fd2)
        
        for j=1:length(Fd2{i}.data)
            FD2data=[FD2data;Fd2{i}.data{j}];
            
            for k=1:length(Fd2{i}.labels{j})
                FD2labels=[FD2labels;Fd2{i}.labels{j}{k}];
            end
        end
        
    end
    
    FD3data=[];
    FD3labels=[];
    
    for i=1:length(Fd3)
        
        for j=1:length(Fd3{i}.data)
            FD3data=[FD3data;Fd3{i}.data{j}];
            
            for k=1:length(Fd3{i}.labels{j})
                FD3labels=[FD3labels;Fd3{i}.labels{j}{k}];
            end
        end
        
    end
    
    mValues=max(FD3data);
    
    % perform PCA
    [coeff,score,eigenvalues]=pca(FD3data./repmat(mValues,size(FD3data,1),1));
    
    plot_sparsity(eigenvalues,score,4,5,FD3labels, 'Distribution of the training data');

    % centralize the data, by substracting the means
    cntrddata1=(FD1data./repmat(mValues,size(FD1data,1),1))-repmat(mean(FD3data./repmat(mValues,size(FD3data,1),1)),size(FD1data,1),1);

    % project the data to the new hyperplane
    FD1_projdata=cntrddata1*coeff;

    
    % centralize the data, by substracting the means
    cntrddata2=(FD2data./repmat(mValues,size(FD2data,1),1))-repmat(mean(FD3data./repmat(mValues,size(FD3data,1),1)),size(FD2data,1),1);
    
    % project the data to the new hyperplane
    FD2_projdata=cntrddata2*coeff;
    
    figure(sbj*10+3)
    
    % fit data to GMMs
    options = statset('MaxIter',1000);
    BIC_cri=struct([]);
    BIC_cri{1}=[];
    BIC_cri{2}=[];
    BIC_cri{3}=[];
    
    AIC_cri=struct([]);
    AIC_cri{1}=[];
    AIC_cri{2}=[];
    AIC_cri{3}=[];
    
    for nb_g=1:10
        gmodel1=fitgmdist(score(:,1:2),nb_g,'Options',options);
        BIC_cri{1}=[BIC_cri{1};gmodel1.BIC];
        AIC_cri{1}=[AIC_cri{1};gmodel1.AIC];
        gmodel2=fitgmdist(FD1_projdata(:,1:2),nb_g,'Options',options);
        BIC_cri{2}=[BIC_cri{2};gmodel2.BIC];
        AIC_cri{2}=[AIC_cri{2};gmodel2.AIC];
        gmodel3=fitgmdist(FD2_projdata(:,1:2),nb_g,'Options',options);
        BIC_cri{3}=[BIC_cri{3};gmodel3.BIC];
        AIC_cri{3}=[AIC_cri{3};gmodel3.AIC];
    end
    
    figure(1000+sbj)
    subplot(3,1,1)
    plot(1:10,BIC_cri{1},'b')
    hold on
    plot(1:10,AIC_cri{1},'r')
    title('model 1')
    subplot(3,1,2)
    plot(1:10,BIC_cri{2},'b')
    hold on
    plot(1:10,AIC_cri{2},'r')
    title('model 2')
    subplot(3,1,3)
    plot(1:10,BIC_cri{3},'b')
    hold on
    plot(1:10,AIC_cri{3},'r')
    title('model 3')
    xlabel('number of Gaussians')
    legend('BIC','AIC')
    
    gmodel1=fitgmdist(score(:,1:2),3,'Options',options);
    gmodel2=fitgmdist(FD1_projdata(:,1:2),4,'Options',options);
    gmodel3=fitgmdist(FD2_projdata(:,1:2),3,'Options',options);
    
    
    % find probability distribution
    d=500;
    
    x1=linspace(min(score(:,1)),max(score(:,1)),d);
    x2=linspace(min(score(:,2)),max(score(:,2)),d);
    [X,Y] = meshgrid(x1,x2);
    tx31=reshape(X,size(X,1)*size(X,2),1);
    tx32=reshape(Y,size(Y,1)*size(Y,2),1);
    y3=pdf(gmodel1,[tx31,tx32]);
    CO(:,:,1)=-0.1-log10(reshape(y3/max(y3),size(Y,1),size(Y,2)));
    CO(:,:,2)=ones(500);
    CO(:,:,3)=-0.1-log10(reshape(y3/max(y3),size(Y,1),size(Y,2)));
    figure(sbj*10+3)
    hold on
    s1=scatter(gmodel2.mu(:,1),gmodel2.mu(:,2),50,ones(size(gmodel2.mu,1),1),'filled','b','MarkerFaceAlpha',0.6);
    s2=scatter(gmodel3.mu(:,1),gmodel3.mu(:,2),50,ones(size(gmodel3.mu,1),1),'filled','r','MarkerFaceAlpha',0.6);
    s3=scatter(gmodel1.mu(:,1),gmodel1.mu(:,2),50,ones(size(gmodel1.mu,1),1),'filled','g','MarkerFaceAlpha',0.6);
    s1.MarkerEdgeColor='blue';
    s2.MarkerEdgeColor='red';
    s3.MarkerEdgeColor='green';
    s1.MarkerFaceColor='blue';
    s2.MarkerFaceColor='red';
    s3.MarkerFaceColor='green';
    legend('1st phase','2nd phase','3rd phase')
%     figure
    hold on
    h=surf(X,Y,reshape(y3/max(y3),size(Y,1),size(Y,2)),CO,'FaceColor','flat','FaceAlpha',0.6,'FaceLighting','none');
    set(h,'LineStyle','none')
    view([0 90])
    
    

%     title('1st phase')
%     grid on
    
    x1=linspace(min(FD2_projdata(:,1))-0.2,max(FD2_projdata(:,1)),d);
    x2=linspace(min(FD2_projdata(:,2))-0.1,max(FD2_projdata(:,2)),d);
    [X,Y] = meshgrid(x1,x2);
    tx21=reshape(X,size(X,1)*size(X,2),1);
    tx22=reshape(Y,size(Y,1)*size(Y,2),1);
    y2=pdf(gmodel3,[tx21,tx22]);
    CO(:,:,1)=ones(500);
    CO(:,:,2)=-0.2-log10(reshape(y2/max(y2),size(Y,1),size(Y,2)));
    CO(:,:,3)=-0.2-log10(reshape(y2/max(y2),size(Y,1),size(Y,2)));
    figure(sbj*10+3)
%     figure
    hold on
    h=surf(X,Y,reshape(y2/max(y2),size(Y,1),size(Y,2)),CO,'FaceColor','flat','FaceAlpha',0.6,'FaceLighting','none');
    set(h,'LineStyle','none')
%     title('2nd phase')
%     grid on
    
    x1=linspace(min(FD1_projdata(:,1)),max(FD1_projdata(:,1)),d);
    x2=linspace(min(FD1_projdata(:,2)),max(FD1_projdata(:,2)),d);
    [X,Y] = meshgrid(x1,x2);
    tx11=reshape(X,size(X,1)*size(X,2),1);
    tx12=reshape(Y,size(Y,1)*size(Y,2),1);
    y1=pdf(gmodel2,[tx11,tx12]);
    CO(:,:,1)=-0.2-log10(reshape(y1/max(y1),size(Y,1),size(Y,2)));
    CO(:,:,2)=-0.2-log10(reshape(y1/max(y1),size(Y,1),size(Y,2)));
    CO(:,:,3)=ones(500);
    figure(sbj*10+3)
%     figure
    hold on
    h=surf(X,Y,reshape(y1/max(y1),size(Y,1),size(Y,2)),CO,'FaceColor','flat','FaceAlpha',0.6,'FaceLighting','none');
    set(h,'LineStyle','none')
%     title('3rd phase')
    
    ezcontour(@(x1,x2)pdf(gmodel1,[x1 x2]),get(gca,{'XLim','YLim'}))
    ezcontour(@(x1,x2)pdf(gmodel2,[x1 x2]),get(gca,{'XLim','YLim'}))
    ezcontour(@(x1,x2)pdf(gmodel3,[x1 x2]),get(gca,{'XLim','YLim'}))
    scatter(gmodel2.mu(:,1),gmodel2.mu(:,2),50,ones(size(gmodel2.mu,1),1),'filled','b','MarkerFaceAlpha',0.2);
    scatter(gmodel3.mu(:,1),gmodel3.mu(:,2),50,ones(size(gmodel3.mu,1),1),'filled','r','MarkerFaceAlpha',0.2);
    scatter(gmodel1.mu(:,1),gmodel1.mu(:,2),50,ones(size(gmodel1.mu,1),1),'filled','g','MarkerFaceAlpha',0.2);
    title(['TR' num2str(sbj) ' - complete muscleset' ])
    xlabel('1^s^t PC')
    ylabel('2^n^d PC')
    zlabel('Density')
    axis([-1 1 -1 1])
%     view([-30 50])
    view([43 56])
    set(gca,'FontSize',35,'FontWeight','bold')
    grid on
    
    
    
    % visualization of the data
    figure(sbj*10+2)
    s1=scatter3(FD1_projdata(1,1),FD1_projdata(1,2),FD1_projdata(1,3),10,1,'magenta');
    hold on
    s2=scatter3(FD2_projdata(1,1),FD2_projdata(1,2),FD2_projdata(1,3),10,1,'red');
    s3=scatter3(score(1,1),score(1,2),score(1,3),10,1,'cyan');
    s1.MarkerEdgeColor='magenta';
    s2.MarkerEdgeColor='red';
    s3.MarkerEdgeColor='cyan';
    s1.MarkerFaceColor='magenta';
    s2.MarkerFaceColor='red';
    s3.MarkerFaceColor='cyan';
    legend('1st phase','2nd phase','3rd phase')
    scatter3(FD1_projdata(:,1),FD1_projdata(:,2),FD1_projdata(:,3),10,ones(length(FD1labels),1),'filled','magenta')
    scatter3(FD2_projdata(:,1),FD2_projdata(:,2),FD2_projdata(:,3),10,ones(length(FD2labels),1),'filled','red')
    scatter3(score(:,1),score(:,2),score(:,3),10,ones(length(FD3labels),1),'filled','cyan')
    grid on
    title(['AMP' num2str(sbj)])
    xlabel('1st PC')
    ylabel('2nd PC')
    zlabel('3rd PC')
    
    figure(sbj*10+8)
    s1=scatter(FD1_projdata(1,1),FD1_projdata(1,2),10,1,'filled','magenta');
    hold on
    s2=scatter(FD2_projdata(1,1),FD2_projdata(1,2),10,1,'filled','red');
    s3=scatter(score(1,1),score(1,2),10,1,'filled','cyan');
    s1.MarkerEdgeColor='magenta';
    s2.MarkerEdgeColor='red';
    s3.MarkerEdgeColor='cyan';
    s1.MarkerFaceColor='magenta';
    s2.MarkerFaceColor='red';
    s3.MarkerFaceColor='cyan';
    legend('1st phase','2nd phase','3rd phase')
    scatter(FD1_projdata(:,1),FD1_projdata(:,2),10,ones(length(FD1labels),1),'filled','magenta')
    scatter(FD2_projdata(:,1),FD2_projdata(:,2),10,ones(length(FD2labels),1),'filled','red')
    scatter(score(:,1),score(:,2),10,ones(length(FD3labels),1),'filled','cyan')
    grid on
    xlabel('1st PC')
    ylabel('2nd PC')
    title(['AMP' num2str(sbj)])
    set(gca,'FontSize',35,'FontWeight','bold')
    
    

%     figure(sbj*10+3)
%     scatter(gmodel2.mu(1),gmodel2.mu(2),100,'b')
%     hold on
%     scatter(gmodel3.mu(1),gmodel3.mu(2),100,'r')
%     scatter(gmodel1.mu(1),gmodel1.mu(2),100,'g')
%     grid on
%     legend('1st phase','2nd phase','3rd phase')
%     
%     
%     tt=y3/max(y3);
%     scatter(tx31(tt>0.05),tx32(tt>0.05),10,[-log10(tt(tt>0.05)),ones(length(tt(tt>0.05)),1),-log10(tt(tt>0.05))],'MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)
%     tt=y2/max(y2);
%     scatter(tx21(tt>0.05),tx22(tt>0.05),10,[ones(length(tt(tt>0.05)),1),-log10(tt(tt>0.05)),-log10(tt(tt>0.05))]);%,'MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)
%     tt=y1/max(y1);
%     scatter(tx11(tt>0.05),tx12(tt>0.05),10,[-log10(tt(tt>0.05)),-log10(tt(tt>0.05)),ones(length(tt(tt>0.05)),1)],'MarkerEdgeAlpha',0.5,'MarkerFaceAlpha',0.5)
%     
%     ezcontour(@(x1,x2)pdf(gmodel1,[x1 x2]),get(gca,{'XLim','YLim'}))
%     ezcontour(@(x1,x2)pdf(gmodel2,[x1 x2]),get(gca,{'XLim','YLim'}))
%     ezcontour(@(x1,x2)pdf(gmodel3,[x1 x2]),get(gca,{'XLim','YLim'}))
    
    figure(sbj*10+4)
    subplot(3,2,1)
    scatter3(score((FD3labels==1),1),score((FD3labels==1),2),score((FD3labels==1),3),100,ones(length(FD3labels(FD3labels==1)),1),'filled','cyan')
    hold on
    scatter3(score((FD1labels==1),1),score((FD1labels==1),2),score((FD1labels==1),3),100,ones(length(FD1labels(FD1labels==1)),1),'filled','magenta')
    scatter3(score((FD2labels==1),1),score((FD2labels==1),2),score((FD2labels==1),3),100,ones(length(FD2labels(FD2labels==1)),1),'filled','red')
    title('precision')
    grid on
    
    subplot(3,2,2)
    scatter3(score((FD3labels==2),1),score((FD3labels==2),2),score((FD3labels==2),3),100,ones(length(FD3labels(FD3labels==2)),1),'filled','cyan')
    hold on
    scatter3(score((FD1labels==2),1),score((FD1labels==2),2),score((FD1labels==2),3),100,ones(length(FD1labels(FD1labels==2)),1),'filled','magenta')
    scatter3(score((FD2labels==2),1),score((FD2labels==2),2),score((FD2labels==2),3),100,ones(length(FD2labels(FD2labels==2)),1),'filled','red')
    title('t2')
    grid on
    
    subplot(3,2,3)
    scatter3(score((FD3labels==3),1),score((FD3labels==3),2),score((FD3labels==3),3),100,ones(length(FD3labels(FD3labels==3)),1),'filled','cyan')
    hold on
    scatter3(score((FD1labels==3),1),score((FD1labels==3),2),score((FD1labels==3),3),100,ones(length(FD1labels(FD1labels==3)),1),'filled','magenta')
    scatter3(score((FD2labels==3),1),score((FD2labels==3),2),score((FD2labels==3),3),100,ones(length(FD2labels(FD2labels==3)),1),'filled','red')
    title('t4')
    grid on
    
    subplot(3,2,4)
    scatter3(score((FD3labels==4),1),score((FD3labels==4),2),score((FD3labels==4),3),100,ones(length(FD3labels(FD3labels==4)),1),'filled','cyan')
    hold on
    scatter3(score((FD1labels==4),1),score((FD1labels==4),2),score((FD1labels==4),3),100,ones(length(FD1labels(FD1labels==4)),1),'filled','magenta')
    scatter3(score((FD2labels==4),1),score((FD2labels==4),2),score((FD2labels==4),3),100,ones(length(FD2labels(FD2labels==4)),1),'filled','red')
    title('pinch')
    grid on
    
    subplot(3,2,5)
    scatter3(score((FD3labels==5),1),score((FD3labels==5),2),score((FD3labels==5),3),100,ones(length(FD3labels(FD3labels==5)),1),'filled','cyan')
    hold on
    scatter3(score((FD1labels==5),1),score((FD1labels==5),2),score((FD1labels==5),3),100,ones(length(FD1labels(FD1labels==5)),1),'filled','magenta')
    scatter3(score((FD2labels==5),1),score((FD2labels==5),2),score((FD2labels==5),3),100,ones(length(FD2labels(FD2labels==5)),1),'filled','red')
    title('lateral')
    grid on
    
    figure(sbj*10+5)
    subplot(3,2,1)
    scatter(score((FD3labels==1),1),score((FD3labels==1),2),100,ones(length(FD3labels(FD3labels==1)),1),'filled','cyan')
    hold on
    scatter(score((FD1labels==1),1),score((FD1labels==1),2),100,ones(length(FD1labels(FD1labels==1)),1),'filled','magenta')
    scatter(score((FD2labels==1),1),score((FD2labels==1),2),100,ones(length(FD2labels(FD2labels==1)),1),'filled','red')
    title('precision')
    grid on
    
    subplot(3,2,2)
    scatter(score((FD3labels==2),1),score((FD3labels==2),2),100,ones(length(FD3labels(FD3labels==2)),1),'filled','cyan')
    hold on
    scatter(score((FD1labels==2),1),score((FD1labels==2),2),100,ones(length(FD1labels(FD1labels==2)),1),'filled','magenta')
    scatter(score((FD2labels==2),1),score((FD2labels==2),2),100,ones(length(FD2labels(FD2labels==2)),1),'filled','red')
    title('t2')
    grid on
    
    subplot(3,2,3)
    scatter(score((FD3labels==3),1),score((FD3labels==3),2),100,ones(length(FD3labels(FD3labels==3)),1),'filled','cyan')
    hold on
    scatter(score((FD1labels==3),1),score((FD1labels==3),2),100,ones(length(FD1labels(FD1labels==3)),1),'filled','magenta')
    scatter(score((FD2labels==3),1),score((FD2labels==3),2),100,ones(length(FD2labels(FD2labels==3)),1),'filled','red')
    title('t4')
    grid on
    
    subplot(3,2,4)
    scatter(score((FD3labels==4),1),score((FD3labels==4),2),100,ones(length(FD3labels(FD3labels==4)),1),'filled','cyan')
    hold on
    scatter(score((FD1labels==4),1),score((FD1labels==4),2),100,ones(length(FD1labels(FD1labels==4)),1),'filled','magenta')
    scatter(score((FD2labels==4),1),score((FD2labels==4),2),100,ones(length(FD2labels(FD2labels==4)),1),'filled','red')
    title('pinch')
    grid on
    
    subplot(3,2,5)
    scatter(score((FD3labels==5),1),score((FD3labels==5),2),100,ones(length(FD3labels(FD3labels==5)),1),'filled','cyan')
    hold on
    scatter(score((FD1labels==5),1),score((FD1labels==5),2),100,ones(length(FD1labels(FD1labels==5)),1),'filled','magenta')
    scatter(score((FD2labels==5),1),score((FD2labels==5),2),100,ones(length(FD2labels(FD2labels==5)),1),'filled','red')
    title('lateral')
    grid on
    
    
    
end



%%


for sbj=2:2
    
    [d1,d2,d3,testingSet,Fd1,Fd2,Fd3,FtestingSet]=createDataSets_byAngle(sbjData{sbj},2:6,15,sbj,'MuscleSet',muscleSets{sbj}{2},'Normalization',true,'FeatureExtraction',features);
    
    FD1data=[];
    FD1labels=[];
    
    for i=1:length(Fd1)
        
        for j=1:length(Fd1{i}.data)
            FD1data=[FD1data;Fd1{i}.data{j}];
            
            for k=1:length(Fd1{i}.labels{j})
                FD1labels=[FD1labels;Fd1{i}.labels{j}{k}];
            end
        end
        
    end
    
    
    FD2data=[];
    FD2labels=[];
    
    for i=1:length(Fd2)
        
        for j=1:length(Fd2{i}.data)
            FD2data=[FD2data;Fd2{i}.data{j}];
            
            for k=1:length(Fd2{i}.labels{j})
                FD2labels=[FD2labels;Fd2{i}.labels{j}{k}];
            end
        end
        
    end
    
    FD3data=[];
    FD3labels=[];
    
    for i=1:length(Fd3)
        
        for j=1:length(Fd3{i}.data)
            FD3data=[FD3data;Fd3{i}.data{j}];
            
            for k=1:length(Fd3{i}.labels{j})
                FD3labels=[FD3labels;Fd3{i}.labels{j}{k}];
            end
        end
        
    end
    
    mValues=max(FD3data);
    
    % perform PCA
    [coeff,score,eigenvalues]=pca(FD3data./repmat(mValues,size(FD3data,1),1));
    
    plot_sparsity(eigenvalues,score,4,5,FD3labels, 'Distribution of the training data');

    % centralize the data, by substracting the means
    cntrddata1=(FD1data./repmat(mValues,size(FD1data,1),1))-repmat(mean(FD3data./repmat(mValues,size(FD3data,1),1)),size(FD1data,1),1);

    % project the data to the new hyperplane
    FD1_projdata=cntrddata1*coeff;

    
    % centralize the data, by substracting the means
    cntrddata2=(FD2data./repmat(mValues,size(FD2data,1),1))-repmat(mean(FD3data./repmat(mValues,size(FD3data,1),1)),size(FD2data,1),1);
    
    % project the data to the new hyperplane
    FD2_projdata=cntrddata2*coeff;
    
    figure(sbj*10+3)
    
    % fit data to GMMs
    options = statset('MaxIter',1000);
    BIC_cri=struct([]);
    BIC_cri{1}=[];
    BIC_cri{2}=[];
    BIC_cri{3}=[];
    
    AIC_cri=struct([]);
    AIC_cri{1}=[];
    AIC_cri{2}=[];
    AIC_cri{3}=[];
    
    for nb_g=1:10
        gmodel1=fitgmdist(score(:,1:2),nb_g,'Options',options);
        BIC_cri{1}=[BIC_cri{1};gmodel1.BIC];
        AIC_cri{1}=[AIC_cri{1};gmodel1.AIC];
        gmodel2=fitgmdist(FD1_projdata(:,1:2),nb_g,'Options',options);
        BIC_cri{2}=[BIC_cri{2};gmodel2.BIC];
        AIC_cri{2}=[AIC_cri{2};gmodel2.AIC];
        gmodel3=fitgmdist(FD2_projdata(:,1:2),nb_g,'Options',options);
        BIC_cri{3}=[BIC_cri{3};gmodel3.BIC];
        AIC_cri{3}=[AIC_cri{3};gmodel3.AIC];
    end
    
    figure(1000+sbj)
    subplot(3,1,1)
    plot(1:10,BIC_cri{1},'b')
    hold on
    plot(1:10,AIC_cri{1},'r')
    title('model 1')
    subplot(3,1,2)
    plot(1:10,BIC_cri{2},'b')
    hold on
    plot(1:10,AIC_cri{2},'r')
    title('model 2')
    subplot(3,1,3)
    plot(1:10,BIC_cri{3},'b')
    hold on
    plot(1:10,AIC_cri{3},'r')
    title('model 3')
    xlabel('number of Gaussians')
    legend('BIC','AIC')
    
    gmodel1=fitgmdist(score(:,1:2),2,'Options',options);
    gmodel2=fitgmdist(FD1_projdata(:,1:2),2,'Options',options);
    gmodel3=fitgmdist(FD2_projdata(:,1:2),2,'Options',options);
    
    
    % find probability distribution
    d=500;
    
    x1=linspace(min(score(:,1))-0.3,max(score(:,1)),d);
    x2=linspace(min(score(:,2))-0.3,max(score(:,2)),d);
    [X,Y] = meshgrid(x1,x2);
    tx31=reshape(X,size(X,1)*size(X,2),1);
    tx32=reshape(Y,size(Y,1)*size(Y,2),1);
    y3=pdf(gmodel1,[tx31,tx32]);
    CO(:,:,1)=-0.1-log10(reshape(y3/max(y3),size(Y,1),size(Y,2)));
    CO(:,:,2)=ones(500);
    CO(:,:,3)=-0.1-log10(reshape(y3/max(y3),size(Y,1),size(Y,2)));
    figure(sbj*10+3)
    hold on
    s1=scatter(gmodel2.mu(:,1),gmodel2.mu(:,2),50,ones(size(gmodel2.mu,1),1),'filled','b','MarkerFaceAlpha',0.9);
    s2=scatter(gmodel3.mu(:,1),gmodel3.mu(:,2),50,ones(size(gmodel3.mu,1),1),'filled','r','MarkerFaceAlpha',0.9);
    s3=scatter(gmodel1.mu(:,1),gmodel1.mu(:,2),50,ones(size(gmodel1.mu,1),1),'filled','g','MarkerFaceAlpha',0.9);
    s1.MarkerEdgeColor='blue';
    s2.MarkerEdgeColor='red';
    s3.MarkerEdgeColor='green';
    s1.MarkerFaceColor='blue';
    s2.MarkerFaceColor='red';
    s3.MarkerFaceColor='green';
    legend('1st phase','2nd phase','3rd phase')
%     figure
    hold on
    h=surf(X,Y,reshape(y3/max(y3),size(Y,1),size(Y,2)),CO,'FaceColor','flat','FaceAlpha',0.95,'FaceLighting','none');
    set(h,'LineStyle','none')
    view([0 90])
    
    

%     title('1st phase')
%     grid on
    
    x1=linspace(min(FD2_projdata(:,1))-0.3,max(FD2_projdata(:,1)),d);
    x2=linspace(min(FD2_projdata(:,2))-0.3,max(FD2_projdata(:,2)),d);
    [X,Y] = meshgrid(x1,x2);
    tx21=reshape(X,size(X,1)*size(X,2),1);
    tx22=reshape(Y,size(Y,1)*size(Y,2),1);
    y2=pdf(gmodel3,[tx21,tx22]);
    CO(:,:,1)=ones(500);
    CO(:,:,2)=-0.2-log10(reshape(y2/max(y2),size(Y,1),size(Y,2)));
    CO(:,:,3)=-0.2-log10(reshape(y2/max(y2),size(Y,1),size(Y,2)));
    figure(sbj*10+3)
%     figure
    hold on
    h=surf(X,Y,reshape(y2/max(y2),size(Y,1),size(Y,2)),CO,'FaceColor','flat','FaceAlpha',0.95,'FaceLighting','none');
    set(h,'LineStyle','none')
%     title('2nd phase')
%     grid on
    
    x1=linspace(min(FD1_projdata(:,1))-0.3,max(FD1_projdata(:,1)),d);
    x2=linspace(min(FD1_projdata(:,2))-0.3,max(FD1_projdata(:,2)),d);
    [X,Y] = meshgrid(x1,x2);
    tx11=reshape(X,size(X,1)*size(X,2),1);
    tx12=reshape(Y,size(Y,1)*size(Y,2),1);
    y1=pdf(gmodel2,[tx11,tx12]);
    CO(:,:,1)=-0.2-log10(reshape(y1/max(y1),size(Y,1),size(Y,2)));
    CO(:,:,2)=-0.2-log10(reshape(y1/max(y1),size(Y,1),size(Y,2)));
    CO(:,:,3)=ones(500);
    figure(sbj*10+3)
%     figure
    hold on
    h=surf(X,Y,reshape(y1/max(y1),size(Y,1),size(Y,2)),CO,'FaceColor','flat','FaceAlpha',0.5,'FaceLighting','none');
    set(h,'LineStyle','none')
%     title('3rd phase')
    
    ezcontour(@(x1,x2)pdf(gmodel1,[x1 x2]),get(gca,{'XLim','YLim'}))
    ezcontour(@(x1,x2)pdf(gmodel2,[x1 x2]),get(gca,{'XLim','YLim'}))
    ezcontour(@(x1,x2)pdf(gmodel3,[x1 x2]),get(gca,{'XLim','YLim'}))
    scatter(gmodel2.mu(:,1),gmodel2.mu(:,2),50,ones(size(gmodel2.mu,1),1),'filled','b','MarkerFaceAlpha',0.2);
    scatter(gmodel3.mu(:,1),gmodel3.mu(:,2),50,ones(size(gmodel3.mu,1),1),'filled','r','MarkerFaceAlpha',0.2);
    scatter(gmodel1.mu(:,1),gmodel1.mu(:,2),50,ones(size(gmodel1.mu,1),1),'filled','g','MarkerFaceAlpha',0.2);
    title(['TR' num2str(sbj) ' - forearm muscles'])
    xlabel('1^s^t PC')
    ylabel('2^n^d PC')
    zlabel('Density')
    axis([-1 1 -1 1])
%     view([-30 50])
    view([43 56])
    grid on
    
    set(gca,'FontSize',35,'FontWeight','bold')

    
    
    % visualization of the data
    figure(sbj*10+2)
    s1=scatter3(FD1_projdata(1,1),FD1_projdata(1,2),FD1_projdata(1,3),10,1,'magenta');
    hold on
    s2=scatter3(FD2_projdata(1,1),FD2_projdata(1,2),FD2_projdata(1,3),10,1,'red');
    s3=scatter3(score(1,1),score(1,2),score(1,3),10,1,'cyan');
    s1.MarkerEdgeColor='magenta';
    s2.MarkerEdgeColor='red';
    s3.MarkerEdgeColor='cyan';
    s1.MarkerFaceColor='magenta';
    s2.MarkerFaceColor='red';
    s3.MarkerFaceColor='cyan';
    legend('1st phase','2nd phase','3rd phase')
    scatter3(FD1_projdata(:,1),FD1_projdata(:,2),FD1_projdata(:,3),10,ones(length(FD1labels),1),'filled','magenta')
    scatter3(FD2_projdata(:,1),FD2_projdata(:,2),FD2_projdata(:,3),10,ones(length(FD2labels),1),'filled','red')
    scatter3(score(:,1),score(:,2),score(:,3),10,ones(length(FD3labels),1),'filled','cyan')
    grid on
    title(['AMP' num2str(sbj)])
    xlabel('1st PC')
    ylabel('2nd PC')
    zlabel('3rd PC')
    
    figure(sbj*10+8)
    s1=scatter(FD1_projdata(1,1),FD1_projdata(1,2),10,1,'filled','magenta');
    hold on
    s2=scatter(FD2_projdata(1,1),FD2_projdata(1,2),10,1,'filled','red');
    s3=scatter(score(1,1),score(1,2),10,1,'filled','cyan');
    s1.MarkerEdgeColor='magenta';
    s2.MarkerEdgeColor='red';
    s3.MarkerEdgeColor='cyan';
    s1.MarkerFaceColor='magenta';
    s2.MarkerFaceColor='red';
    s3.MarkerFaceColor='cyan';
    legend('1st phase','2nd phase','3rd phase')
    scatter(FD1_projdata(:,1),FD1_projdata(:,2),10,ones(length(FD1labels),1),'filled','magenta')
    scatter(FD2_projdata(:,1),FD2_projdata(:,2),10,ones(length(FD2labels),1),'filled','red')
    scatter(score(:,1),score(:,2),10,ones(length(FD3labels),1),'filled','cyan')
    grid on
    xlabel('1st PC')
    ylabel('2nd PC')
    title(['AMP' num2str(sbj)])
    
    

%     figure(sbj*10+3)
%     scatter(gmodel2.mu(1),gmodel2.mu(2),100,'b')
%     hold on
%     scatter(gmodel3.mu(1),gmodel3.mu(2),100,'r')
%     scatter(gmodel1.mu(1),gmodel1.mu(2),100,'g')
%     grid on
%     legend('1st phase','2nd phase','3rd phase')
%     
%     
%     tt=y3/max(y3);
%     scatter(tx31(tt>0.05),tx32(tt>0.05),10,[-log10(tt(tt>0.05)),ones(length(tt(tt>0.05)),1),-log10(tt(tt>0.05))],'MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)
%     tt=y2/max(y2);
%     scatter(tx21(tt>0.05),tx22(tt>0.05),10,[ones(length(tt(tt>0.05)),1),-log10(tt(tt>0.05)),-log10(tt(tt>0.05))]);%,'MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)
%     tt=y1/max(y1);
%     scatter(tx11(tt>0.05),tx12(tt>0.05),10,[-log10(tt(tt>0.05)),-log10(tt(tt>0.05)),ones(length(tt(tt>0.05)),1)],'MarkerEdgeAlpha',0.5,'MarkerFaceAlpha',0.5)
%     
%     ezcontour(@(x1,x2)pdf(gmodel1,[x1 x2]),get(gca,{'XLim','YLim'}))
%     ezcontour(@(x1,x2)pdf(gmodel2,[x1 x2]),get(gca,{'XLim','YLim'}))
%     ezcontour(@(x1,x2)pdf(gmodel3,[x1 x2]),get(gca,{'XLim','YLim'}))
    
    figure(sbj*10+4)
    subplot(3,2,1)
    scatter3(score((FD3labels==1),1),score((FD3labels==1),2),score((FD3labels==1),3),100,ones(length(FD3labels(FD3labels==1)),1),'filled','cyan')
    hold on
    scatter3(score((FD1labels==1),1),score((FD1labels==1),2),score((FD1labels==1),3),100,ones(length(FD1labels(FD1labels==1)),1),'filled','magenta')
    scatter3(score((FD2labels==1),1),score((FD2labels==1),2),score((FD2labels==1),3),100,ones(length(FD2labels(FD2labels==1)),1),'filled','red')
    title('precision')
    grid on
    
    subplot(3,2,2)
    scatter3(score((FD3labels==2),1),score((FD3labels==2),2),score((FD3labels==2),3),100,ones(length(FD3labels(FD3labels==2)),1),'filled','cyan')
    hold on
    scatter3(score((FD1labels==2),1),score((FD1labels==2),2),score((FD1labels==2),3),100,ones(length(FD1labels(FD1labels==2)),1),'filled','magenta')
    scatter3(score((FD2labels==2),1),score((FD2labels==2),2),score((FD2labels==2),3),100,ones(length(FD2labels(FD2labels==2)),1),'filled','red')
    title('t2')
    grid on
    
    subplot(3,2,3)
    scatter3(score((FD3labels==3),1),score((FD3labels==3),2),score((FD3labels==3),3),100,ones(length(FD3labels(FD3labels==3)),1),'filled','cyan')
    hold on
    scatter3(score((FD1labels==3),1),score((FD1labels==3),2),score((FD1labels==3),3),100,ones(length(FD1labels(FD1labels==3)),1),'filled','magenta')
    scatter3(score((FD2labels==3),1),score((FD2labels==3),2),score((FD2labels==3),3),100,ones(length(FD2labels(FD2labels==3)),1),'filled','red')
    title('t4')
    grid on
    
    subplot(3,2,4)
    scatter3(score((FD3labels==4),1),score((FD3labels==4),2),score((FD3labels==4),3),100,ones(length(FD3labels(FD3labels==4)),1),'filled','cyan')
    hold on
    scatter3(score((FD1labels==4),1),score((FD1labels==4),2),score((FD1labels==4),3),100,ones(length(FD1labels(FD1labels==4)),1),'filled','magenta')
    scatter3(score((FD2labels==4),1),score((FD2labels==4),2),score((FD2labels==4),3),100,ones(length(FD2labels(FD2labels==4)),1),'filled','red')
    title('pinch')
    grid on
    
    subplot(3,2,5)
    scatter3(score((FD3labels==5),1),score((FD3labels==5),2),score((FD3labels==5),3),100,ones(length(FD3labels(FD3labels==5)),1),'filled','cyan')
    hold on
    scatter3(score((FD1labels==5),1),score((FD1labels==5),2),score((FD1labels==5),3),100,ones(length(FD1labels(FD1labels==5)),1),'filled','magenta')
    scatter3(score((FD2labels==5),1),score((FD2labels==5),2),score((FD2labels==5),3),100,ones(length(FD2labels(FD2labels==5)),1),'filled','red')
    title('lateral')
    grid on
    
    figure(sbj*10+5)
    subplot(3,2,1)
    scatter(score((FD3labels==1),1),score((FD3labels==1),2),100,ones(length(FD3labels(FD3labels==1)),1),'filled','cyan')
    hold on
    scatter(score((FD1labels==1),1),score((FD1labels==1),2),100,ones(length(FD1labels(FD1labels==1)),1),'filled','magenta')
    scatter(score((FD2labels==1),1),score((FD2labels==1),2),100,ones(length(FD2labels(FD2labels==1)),1),'filled','red')
    title('precision')
    grid on
    
    subplot(3,2,2)
    scatter(score((FD3labels==2),1),score((FD3labels==2),2),100,ones(length(FD3labels(FD3labels==2)),1),'filled','cyan')
    hold on
    scatter(score((FD1labels==2),1),score((FD1labels==2),2),100,ones(length(FD1labels(FD1labels==2)),1),'filled','magenta')
    scatter(score((FD2labels==2),1),score((FD2labels==2),2),100,ones(length(FD2labels(FD2labels==2)),1),'filled','red')
    title('t2')
    grid on
    
    subplot(3,2,3)
    scatter(score((FD3labels==3),1),score((FD3labels==3),2),100,ones(length(FD3labels(FD3labels==3)),1),'filled','cyan')
    hold on
    scatter(score((FD1labels==3),1),score((FD1labels==3),2),100,ones(length(FD1labels(FD1labels==3)),1),'filled','magenta')
    scatter(score((FD2labels==3),1),score((FD2labels==3),2),100,ones(length(FD2labels(FD2labels==3)),1),'filled','red')
    title('t4')
    grid on
    
    subplot(3,2,4)
    scatter(score((FD3labels==4),1),score((FD3labels==4),2),100,ones(length(FD3labels(FD3labels==4)),1),'filled','cyan')
    hold on
    scatter(score((FD1labels==4),1),score((FD1labels==4),2),100,ones(length(FD1labels(FD1labels==4)),1),'filled','magenta')
    scatter(score((FD2labels==4),1),score((FD2labels==4),2),100,ones(length(FD2labels(FD2labels==4)),1),'filled','red')
    title('pinch')
    grid on
    
    subplot(3,2,5)
    scatter(score((FD3labels==5),1),score((FD3labels==5),2),100,ones(length(FD3labels(FD3labels==5)),1),'filled','cyan')
    hold on
    scatter(score((FD1labels==5),1),score((FD1labels==5),2),100,ones(length(FD1labels(FD1labels==5)),1),'filled','magenta')
    scatter(score((FD2labels==5),1),score((FD2labels==5),2),100,ones(length(FD2labels(FD2labels==5)),1),'filled','red')
    title('lateral')
    grid on
    
    
    
end