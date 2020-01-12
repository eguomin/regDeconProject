function img2 = align_size_edgepad(img1,Sx2,Sy2,Sz2,padValue)
if(nargin == 4)
    padValue = 0;
end

[Sy1,Sx1,Sz1] = size(img1);
Sx = max(Sx1,Sx2);
Sy = max(Sy1,Sy2);
Sz = max(Sz1,Sz2);
img2 = ones(Sy,Sx,Sz)*padValue;
img2(1:Sy1, 1:Sx1, 1:Sz1)= img1; 

end

