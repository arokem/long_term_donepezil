%function to_plot_d=motion_th_analyze_3(subject, date, session)

function to_plot_d=motion_th_analyze_3(subject, date, session)

%try
%    cd /Applications/MATLAB71/toolbox/matlab/michael_silver/Results
%catch
%    cd ('/Volumes/Macintosh HD-1/Applications/MATLAB71/toolbox/matlab/michael_silver/Results')
%end

load (['motion_th' subject date '_' num2str(session) '.mat'])

to_plot_ht=zeros(1,length(results(1).stimParams.dotDirections));
to_plot_ms=zeros(1,length(results(1).stimParams.dotDirections));
to_plot_cr=zeros(1,length(results(1).stimParams.dotDirections));
to_plot_fa=zeros(1,length(results(1).stimParams.dotDirections));

to_plot_dim=size(to_plot_ht);

for orient=1:to_plot_dim(2)
    for trial=1:length(results(orient).scanHistory.dir1)
        if (results(orient).scanHistory.dir1(trial)==results(orient).scanHistory.dir2(trial))
            if results(orient).scanHistory.response(trial)==2
                to_plot_cr(orient)=to_plot_cr(orient)+1;
            else
                to_plot_fa(orient)=to_plot_fa(orient)+1;
            end
        else
            if results(orient).scanHistory.response(trial)==1
                to_plot_ht(orient)=to_plot_ht(orient)+1;
            else
                to_plot_ms(orient)=to_plot_ms(orient)+1;
            end

        end
    end
end

pre_pre_to_plot = d_prime(to_plot_ht,to_plot_ms,to_plot_fa,to_plot_cr); %calculate dprime
pre_to_plot = [results(1).stimParams.dotDirections; pre_pre_to_plot];

if var(results(1).stimParams.dotDirections)
    to_plot_d=(sortrows(pre_to_plot'))';
else
    to_plot_d=pre_to_plot;
end

plot(to_plot_d(1,:),to_plot_d(2,:),'.-');
set(gca,'XTick',[0 45 90 135 180 225 270 315]);
obliqueness = ((to_plot_d(2,1)+to_plot_d(2,3)+to_plot_d(2,5)+to_plot_d(2,7))-(to_plot_d(2,2)+to_plot_d(2,4)+to_plot_d(2,6)+to_plot_d(2,8)))/sum(to_plot_d(2,:))
mean_d_prime= mean(to_plot_d(2,:))