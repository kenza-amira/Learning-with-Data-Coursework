%
% Versin 0.9  (HS 06/03/2020)
%
% template script for task2_plot_regions_hNN_A.m
% This code was mostly taken from lab 4

% Colormap we will use to colour each classes. 
cmap = autumn(2);

% Generating points
Xplot = linspace(0, 5,150)'; 
Yplot = linspace(0, 5,150)'; 

% Obtain the grid vectors for the two dimensions 
[Xv, Yv] = meshgrid(Xplot, Yplot); 
gridX = [Xv(:), Yv(:)]; % Concatenate to get a 2-D point.

%Doing the classification using the previous task
classes = length(Xv(:)); 
for i = 1:length(gridX)
    classes(i) = task2_hNN_A(gridX(i,:)); 
end

figure; % This function will draw the decision boundaries 
[CC,h] = contourf(Xplot(:), Yplot(:), reshape(classes, length(Xplot ), length(Yplot))); 
set(h,'LineColor','none'); 
colormap(cmap);







