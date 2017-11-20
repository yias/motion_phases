function [output]=createTW2(firstSample,finalSample,Signal,lenthgTW,SR,MuscleSet,overlap,bandPassCuttOffFreq,lowPassCutOffFreq,normaLize,rectiFy,mvc,extFeatures,featuresIDs)

% it discards the time windows that have smaller length than the desired
% length of the time window


output=struct([]);

%Signal=preprocessSignals(Signal(fistSample:end,:),SR,[50,400],20);

% Signal=preprocessSignals(Signal,SR,bandPassCuttOffFreq,lowPassCutOffFreq,normaLize,rectiFy,mvc);

l_TW=lenthgTW*SR;
delay_TW=l_TW-overlap*SR;
countTW=1;


% cut-off frequencies of the band-pass filter for the EMG signals
        
bandPassCuttOffFreq=[50,400];
        
% cut-off frequency of the low-pass filter for the EMG signals
        
lowPassCutOffFreq=20;
        
% order of the filter
filter_order=7;
        
% compute the transfer function coefficients for the EMG filtering
        
Wn=(bandPassCuttOffFreq(1)*2)/SR;
                
[B_H,A_H] = butter(filter_order,Wn,'high'); 

Wn=(bandPassCuttOffFreq(2)*2)/SR;
        
[B_L1,A_L1] = butter(filter_order,Wn,'low'); 

Wn=(lowPassCutOffFreq*2)/SR;
        
[B_L2,A_L2] = butter(filter_order,Wn); 



% history of the emg signals to keep
emgHistory=zeros(round(SR*lenthgTW),length(MuscleSet));

for i=firstSample:delay_TW:finalSample
    
    if i+l_TW<finalSample
        
        emgSignals=OnlinePreprocEMG([emgHistory;Signal(i:i+l_TW,MuscleSet)],SR,B_H,A_H,B_L1,A_L1,B_L2,A_L2,normaLize,rectiFy,mvc,lenthgTW);
        
%         plot(emgSignals)
        
        emgHistory=Signal(i:i+l_TW,MuscleSet);
        if extFeatures
            output{countTW}=exctractFeatures(emgSignals,featuresIDs);
        else
            output{countTW}=emgSignals;
        end
        
         countTW=countTW+1;
    end
    
   
end






end