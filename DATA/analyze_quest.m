
function to_plot=analyze_quest(subject,experiment, date, session)

%function to_plot=analyze_quest(subject,experiment, date, session)
close all
figure

if findstr(experiment,'motion_th')
    
    load (['motion_th' subject date '_' num2str(session) '.mat'])

    results(1).stimParams
    maxTheta=results(1).stimParams.maxTheta;
    to_plot=[results(1).stimParams.dotDirections; maxTheta.*10.^[results(1).scanHistory.t results(2).scanHistory.t results(3).scanHistory.t results(4).scanHistory.t results(5).scanHistory.t results(6).scanHistory.t results(7).scanHistory.t results(8).scanHistory.t]]

    if var(to_plot(1,:))

        to_plot=(sortrows(to_plot'))'
        obliqueness = (mean(to_plot(2,2)+to_plot(2,4)+to_plot(2,6)+to_plot(2,8))-mean(to_plot(2,1)+to_plot(2,3)+to_plot(2,5)+to_plot(2,7)))/sum(to_plot(2,:))

    end

    plot(to_plot(1,:),to_plot(2,:),'.-');
    set(gca,'XTick',[0 45 90 135 180 225 270 315]);
    mean_th= mean(to_plot(2,:))

    figure

    for i=1:8

        if mod(i,2)
            plot_color='r';
        else
            plot_color='b';
        end

        plot(maxTheta.*10.^results(i).scanHistory.q.intensity,strcat('.-',plot_color))
        hold on
    end


elseif findstr(experiment,'VarLearningMotion')
    
    load (['VarLearningMotion' subject date '_' num2str(session) '.mat'])

    results(1).stimParams
    maxTheta=results(1).stimParams.maxTheta;
    to_plot=[results(1).stimParams.dotDirections; maxTheta.*10.^[results(1).scanHistory.t results(2).scanHistory.t results(3).scanHistory.t results(4).scanHistory.t results(5).scanHistory.t results(6).scanHistory.t results(7).scanHistory.t results(8).scanHistory.t]]

    if var(to_plot(1,:))

        to_plot=(sortrows(to_plot'))'
        obliqueness = (mean(to_plot(2,2)+to_plot(2,4)+to_plot(2,6)+to_plot(2,8))-mean(to_plot(2,1)+to_plot(2,3)+to_plot(2,5)+to_plot(2,7)))/sum(to_plot(2,:))

    end

    plot(to_plot(1,:),to_plot(2,:),'.-');
    set(gca,'XTick',[0 45 90 135 180 225 270 315]);
    mean_th= mean(to_plot(2,:))

    figure

    for i=1:8

        if mod(i,2)
            plot_color='r';
        else
            plot_color='b';
        end

        plot(maxTheta.*10.^results(i).scanHistory.q.intensity,strcat('.-',plot_color))
        hold on
    end


elseif findstr(experiment,'texture')



    load (['texture_discrimination' subject date '_' num2str(session) '.mat'])

    results(1).stimParams
    maxTheta=results(1).stimParams.maxTheta

    to_plot=[results(1).stimParams.orientArray; maxTheta.*10.^[results(1).scanHistory.t results(2).scanHistory.t results(3).scanHistory.t results(4).scanHistory.t results(5).scanHistory.t results(6).scanHistory.t results(7).scanHistory.t results(8).scanHistory.t]]

    if var(to_plot(1,:))

        to_plot=(sortrows(to_plot'))'
        obliqueness = (mean(to_plot(2,1)+to_plot(2,2)+to_plot(2,3)+to_plot(2,5)+to_plot(2,6)+to_plot(2,7))-mean(to_plot(2,4)+to_plot(2,8)))/sum(to_plot(2,:))

    end

    plot(to_plot(1,:),to_plot(2,:),'.-');
    set(gca,'XTick',[0 45 90 135 180 225 270 315]);
    mean_th= mean(to_plot(2,:))

    figure
    for i=1:8

        if i==4 | i==8
            plot_color='r';
        elseif i==2 | i==6
            plot_color='b';
        else
            plot_color='g';
        end

        plot(maxTheta*10.^results(i).scanHistory.q.intensity,strcat('.-',plot_color))
        hold on
    end



else
    disp('No such Experiment!')


end

disp(['location:' num2str(results(1).stimParams.locat)])