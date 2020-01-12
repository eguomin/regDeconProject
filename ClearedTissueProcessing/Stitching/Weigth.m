
function W = Weigth(nsize, top, bottom, left, right)
% nsize is the image size as [ny, nx] 
% overlap_top is the overlap in top of the image, y-off, x-off
% W = ones(nsize(1), nsize(2));

% W = ones(nsize(1), nsize(2));
% W(1:top(1),:)= 0;
% W(end-bottom(1)+1:end,:)=0;
% W(:,1:left(1))= 0;
% W(:,end-right(1)+1:end)=0;

top_overlap = linspace(0, 1, top(2));
top_crop = linspace(0, 0, top(1)); 
bottom_overlap = linspace(1, 0, bottom(2));
bottom_crop = linspace(0, 0, bottom(1));
 
center = ones(1,nsize(1) - length(top_overlap) - length(top_crop) - length(bottom_crop) - length(bottom_overlap));
vertial_line = [top_crop, top_overlap, center, bottom_overlap, bottom_crop]';
 
Wy = repmat(vertial_line, [1,nsize(2)]);
  
left_overlap = linspace(0, 1, left(2));
left_crop = linspace(0, 0, left(1));
right_overlap = linspace(1, 0, right(2));
right_crop = linspace(0, 0, right(1));
center = ones(1, nsize(2) - length(left_crop) - length(left_overlap) - length(right_crop) - length(right_overlap));
horizontal_line = [left_crop left_overlap, center, right_overlap, right_crop];
Wx = repmat(horizontal_line, [nsize(1),1]);



W = Wx.*Wy;
%W = imgaussian(W,5);
% WriteTifStack(W, 'K:\StitcherTest\Stomach\b\W.tif','32');
% WriteTifStack(W1, 'K:\StitcherTest\Stomach\b\W1.tif','32');
% WriteTifStack(W1,'K:\StitcherTest\Stomach\b\Wx1.tif','32');
%WriteTifStack(Wy2,'K:\StitcherTest\Stomach\b\Wy2.tif','32');
     
end