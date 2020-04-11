load dset.mat;

task1_1(X,Y_species);

load t1_S.mat;
load t1_R.mat;

disp('The covariance matrix is');
disp(S);

disp('The correlation matrix is');
disp(R);
% for i = 1:24
%     plot(R(:,i));
% end
%for easier color assigning
imagesc(R);
labelnames = {'Feature1','Feature6','Feature11', 'Feature16', 'Feature21',
    'Feature2','Feature7','Feature12', 'Feature17', 'Feature22',
    'Feature3','Feature8','Feature13', 'Feature18', 'Feature23',
    'Feature4','Feature9','Feature14', 'Feature19', 'Feature24',
    'Feature5','Feature10','Feature15', 'Feature20','Feature25'};
set(gca, 'XTick', 1:24); 
set(gca, 'YTick', 1:24); 
set(gca, 'XTickLabel', labelnames); 
xtickangle(90);
set(gca, 'YTickLabel', labelnames); 
title('Correlation Matrix Graph for X', 'FontSize', 14); 
c = jet(10);
caxis([-1;1]);
colormap(c); % set the colorscheme
colorbar;
