function [] = Plot_bilinear(solData)
    
    coord = solData.coord;
    connect = solData.connect;
    u = solData.u;
    f = solData.p;
    
    nEle = size(connect,1);
    
    set(0,'Units','pixels') 
    scnsize = get(0,'ScreenSize');

    pos = [0,...              %linker Rand
           scnsize(4)/1.2,... %unterer Rand
           scnsize(3)/2.5,... %breite
           scnsize(4)/1.5];   %hoehe
    
    % Temperature plot   
       
    % Setting the parameters for the figure
    f1 = figure('Position',pos,'color',[1 1 1],'name', 'Temperature');
    set(gca,'fontsize',14,'FontName','Helvetica')
    hold on
    title(['Temperature']);
    PlotTemp(coord,connect,u,nEle);

%     print(f1, '-dpdf', 'OutputData/Temperature');

    % Heat flux plot
    
    % Setting the parameters for the figure
    f1 = figure('Position',pos,'color',[1 1 1],'name', 'Heat flux');
    set(gca,'fontsize',14,'FontName','Helvetica')
    
    % Heat flux in x-direction subplot
    subplot(2,1,1);
    set(gca,'fontsize',14,'FontName','Helvetica')
    hold on
    title(['Heat flux q_{x}']);
    PlotFlux(coord,connect,f,nEle,1);
    
    % Heat flux in y-direction subplot
    subplot(2,1,2);
    set(gca,'fontsize',14,'FontName','Helvetica')
    hold on
    title(['Heat flux q_{y}'])
    PlotFlux(coord,connect,f,nEle,2);

%     print(f1, '-dpdf', 'OutputData/Heat_flux');

% -------------------------------------------------
% Plot routines:
% -------------------------------------------------
function [] = PlotTemp(coord,connect,u,nEle)
    for i = 1 : nEle;
        xcoor =  [  coord(connect(i,1),1), ...
                    coord(connect(i,2),1), ...
                    coord(connect(i,3),1), ...
                    coord(connect(i,4),1)];
                
        ycoor =  [  coord(connect(i,1),2), ...
                    coord(connect(i,2),2), ...
                    coord(connect(i,3),2), ...
                    coord(connect(i,4),2)]; 

        temp  =  [  u(connect(i,1)), ...
                    u(connect(i,2)), ...
                    u(connect(i,3)), ...       
                    u(connect(i,4))  ...
                 ];  

        % plot
         patch(xcoor,ycoor, temp)
         %axis equal
         axis([0,0.5,0,0.25,0,30]);
        
        % style
        shading interp;
        hold on;
        box on;
        
        
        
    end; 

function [] = PlotFlux(coord,connect,f,nEle,k)
    for i = 1 : nEle;
        xcoor =  [  coord(connect(i,1),1), ...
                    coord(connect(i,2),1), ...
                    coord(connect(i,3),1), ...
                    coord(connect(i,4),1)];
                
        ycoor =  [  coord(connect(i,1),2), ...
                    coord(connect(i,2),2), ...
                    coord(connect(i,3),2), ...
                    coord(connect(i,4),2)]; 
        
        flux  =  [  f(k,i)  , ...
                    f(k+2,i), ...
                    f(k+4,i), ...
                    f(k+6,i)  ...
                 ];
             
        % plot
        patch(xcoor, ycoor, flux)
        %axis equal
        axis([0,0.5,0,0.25]);
        
        % style
        shading interp;
        hold on;
        box on;
    end; 
