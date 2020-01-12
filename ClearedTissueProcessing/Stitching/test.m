clear all;

%W = Weigth_Image([500 500], [100,0], [100,0], [0,100],[0,100]);

% dataA = single(ReadTifStack('K:\StitcherTest\Stomach\Down_Y1_Z1_C1.tif'));
% %dataB = single(ReadTifStack('K:\StitcherTest\b.tif'));
% % tic
% dataB = imtranslate(dataA,[50.3, 0, 50]); %, 'OutputView','full')); %, 'FillValues', mean(dataB(:)))); 
% 
% tic
% [M, Overlap, ce] = Phasor(dataB, dataA, 0, 1);
% toc
%  M
% Overlap
% ce

WriteTifStack(W,'K:\StitcherTest\W.tif','32');
