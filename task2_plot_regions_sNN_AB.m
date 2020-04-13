%
% Versin 0.9  (HS 06/03/2020)
%
% template script for task2_plot_regions_sNN_AB
% Colormap we will use to colour each classes. 
cmap = autumn(2);

% Generating points
Xplot = linspace(-3, 7,200)'; 
Yplot = linspace(-2, 7,200)'; 

% Obtain the grid vectors for the two dimensions 
[Xv, Yv] = meshgrid(Xplot, Yplot); 
gridX = [Xv(:), Yv(:)]; % Concatenate to get a 2-D point.

%Doing the classification using the previous task
classes = length(Xv(:)); 
for i = 1:length(gridX)
    classes(i) = task2_sNN_AB(gridX(i,:)); 
end

figure; % This function will draw the decision boundaries 
[CC,h] = contourf(Xplot(:), Yplot(:), reshape(classes, length(Xplot ), length(Yplot))); 
set(h,'LineColor','none'); 
colormap(cmap);
title('Decision Regions for Task 2.9');
xlabel('X');
ylabel('Y');
saveas(gcf,'t2_regions_sNN_AB.pdf');


