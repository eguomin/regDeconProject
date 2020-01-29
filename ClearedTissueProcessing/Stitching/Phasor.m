function [Matrix, Overlap, ce, RegB] = Phasor(dataB, dataA, GPU_Flag, SubPixelFlag)

if nargin <=2
    GPU_Flag = 0;  %CPU is default mode
end
    
dataA = single(dataA);
dataB = single(dataB);
[sy, sx, sz] = size(dataA);
sizeB0 = size(dataB);
dataB_1 = zeros([sy, sx, sz]);
sizeN =  min([sy, sx, sz], sizeB0);
dataB_1(1:sizeN(1), 1:sizeN(2), 1:sizeN(3))= dataB(1:sizeN(1), 1:sizeN(2), 1:sizeN(3));
dataA = max(dataA,0.001);
dataB_1 = max(dataB_1,0.001);

if GPU_Flag == 1
    g = gpuDevice(1);
    g.FreeMemory;
    FT_AB = fftn(log(gpuArray(dataA))).* conj(fftn(log(gpuArray(dataB_1))));
    Phasor = gather(abs(ifftshift(ifftn(FT_AB./(abs(FT_AB))))));    
else 
    FT_AB = fftn(log(dataA)).* conj(fftn(log(dataB_1)));
    Phasor = abs(ifftshift(ifftn(FT_AB./(abs(FT_AB))))); 
end
    
Phasor(round(sy/2)-2:round(sy/2)+2,round(sx/2)-2:round(sx/2)+2,:) = 0;

%WriteTifStack(single(Phasor),'k:\Phasor.tif','32');

PhasorMax = max(Phasor(:));
I = find(Phasor == PhasorMax);
[Py, Px, Pz] = ind2sub(size(Phasor),I);

% to get sub pixel shift; more info about sub pixel localization: https://en.wikipedia.org/wiki/Phase_correlation
if SubPixelFlag == 1
    SubPhasor = Phasor(Py-2:Py+2,Px-2:Px+2,Pz-2:Pz+2);
    XY = max(SubPhasor,[],3);
    curve = XY(3,:);
    line=1:5;
    line1=1:0.1:5;
    curve1 = interp1(line, curve, line1, 'spline');
    [f, s, mu] = polyfit(line1,curve1,5);
    y = polyval(f,line1,[],mu);
    index = find(y==max(y));
    index = find(curve1==max(curve1));
    Px2 = (index(1)-1)/10-2;

    curve = XY(:,3);
    line=1:5;
    line1=1:0.1:5;
    curve1 = interp1(line, curve, line1,'spline');
    [f, s, mu] = polyfit(line1,curve1,5);
    y = polyval(f,line1,[],mu);
    index = find(y==max(y));
    Py2 = (index(1)-1)/10 -2;
%     plot(line1,curve1,'r*');
%     hold on;
%     plot(line1, y, 'o-');
    ZY = squeeze(max(SubPhasor,[],1));
    curve = ZY(3,:);
    line=1:5;
    line1=1:0.1:5;
    curve1 = interp1(line, curve, line1,'spline');
    [f, s, mu] = polyfit(line1,curve1,5);
    y = polyval(f,line1,[],mu);
    index = find(y==max(y));
    Pz2 = (index(1)-1)/10 -2;
end

CoarseTraslation = size(Phasor)/2 - [Py, Px, Pz] + 1;
a = round(CoarseTraslation);
b = round(CoarseTraslation)-[sy, sx, sz];
p{1} = [a(1), a(2), a(3)];
p{2} = [a(1), a(2), b(3)];
p{3} = [a(1), b(2), a(3)];
p{4} = [a(1), b(2), b(3)];
p{5} = [b(1), a(2), a(3)];
p{6} = [b(1), a(2), b(3)];
p{7} = [b(1), b(2), a(3)];
p{8} = [b(1), b(2), b(3)];

if GPU_Flag == 1
    stackA_gpu = gpuArray(dataA);
    stackB_gpu = gpuArray(dataB_1);
    for c = 1: 8
    %dataB_2 = imtranslate(dataB_1,[-p{c}(2), -p{c}(1), -p{c}(3)]); %, 'OutputView','full')); %, 'FillValues', mean(dataB(:)))); 
    dataB_2 = circshift(stackB_gpu,[-p{c}(1), -p{c}(2), -p{c}(3)]); %, 'OutputView','full')); %, 'FillValues', mean(dataB(:)))); 
    shiftB=[];
    shiftA=[];
    if p{c}(1)>0 & p{c}(1)<=sy & p{c}(2)>0 & p{c}(2)<=sx & p{c}(3)>0 & p{c}(3)<=sz % > > > a a a
          shiftB = dataB_2(1:end-p{c}(1), 1:end-p{c}(2),1:end-p{c}(3));
          shiftA = stackA_gpu(1:end-p{c}(1), 1:end-p{c}(2),1:end-p{c}(3));      
    elseif p{c}(1)<=0 & p{c}(1)>-sy & p{c}(2)<=0 & p{c}(2)>-sx & p{c}(3)<=0 & p{c}(3)>-sz % < < < b b b         
          shiftB = dataB_2(-p{c}(1)+1:end,-p{c}(2)+1:end,-p{c}(3)+1:end);
          shiftA = stackA_gpu(-p{c}(1)+1:end,-p{c}(2)+1:end,-p{c}(3)+1:end);  
    elseif p{c}(1)>0 & p{c}(1)<=sy & p{c}(2)<=0 & p{c}(2)>-sx & p{c}(3)<=0 & p{c}(3)>-sz % > < < a b b
        shiftB = dataB_2(1:end-p{c}(1),-p{c}(2)+1:end, -p{c}(3)+1:end);
        shiftA = stackA_gpu(1:end-p{c}(1),-p{c}(2)+1:end, -p{c}(3)+1:end);
    elseif p{c}(1)<=0 & p{c}(1)>-sy & p{c}(2)>0 & p{c}(2)<=sx & p{c}(3)>0 & p{c}(3)<=sz % < > > b a a
        shiftB = dataB_2(-p{c}(1)+1:end,1:end-p{c}(2),1:end-p{c}(3));
        shiftA = stackA_gpu(-p{c}(1)+1:end,1:end-p{c}(2),1:end-p{c}(3));
     elseif p{c}(1)>0 & p{c}(1)<=sy & p{c}(2)>0 & p{c}(2)<=sx & p{c}(3)<=0 & p{c}(3)>-sz % > > < a a b
         shiftB = dataB_2(1:end-p{c}(1),1:end-p{c}(2),-p{c}(3)+1:end);
         shiftA = stackA_gpu(1:end-p{c}(1),1:end-p{c}(2),-p{c}(3)+1:end);
    elseif p{c}(1)<=0 & p{c}(1)>-sy & p{c}(2)>0 & p{c}(2)<=sx & p{c}(3)<=0 & p{c}(3)>-sz% < > < b a b
        shiftB = dataB_2(-p{c}(1)+1:end,1:end-p{c}(2),-p{c}(3)+1:end);
        shiftA = stackA_gpu(-p{c}(1)+1:end,1:end-p{c}(2),-p{c}(3)+1:end);
    elseif p{c}(1)>0 & p{c}(1)<=sy & p{c}(2)<=0 & p{c}(2)>-sx & p{c}(3)>0 & p{c}(3)<=sz  % > < > a b a
        shiftB = dataB_2(1:end-p{c}(1),-p{c}(2)+1:end,1:end-p{c}(3));
        shiftA = stackA_gpu(1:end-p{c}(1),-p{c}(2)+1:end,1:end-p{c}(3));
    elseif p{c}(1)<=0 & p{c}(1)>-sy & p{c}(2)<=0 & p{c}(2)>-sx & p{c}(3)>0 & p{c}(3)<=sz % < < > b b a
        shiftB = dataB_2(-p{c}(1)+1:end, -p{c}(2)+1:end, 1:end-p{c}(3));
        shiftA = stackA_gpu(-p{c}(1)+1:end, -p{c}(2)+1:end, 1:end-p{c}(3));
    end
    
    shiftA = shiftA - mean(shiftA(:));
    shiftB = shiftB - mean(shiftB(:));
    Coef = shiftA.*shiftB/(std(shiftA(:))*std(shiftB(:)));
    [ny, nx, nz] = size(shiftA);
    r(c) = sum(Coef(:))/(ny*nx*nz);
    
    end
end

if GPU_Flag == 0
    for c = 1: 8
    %dataB_2 = imtranslate(dataB_1,[-p{c}(2), -p{c}(1), -p{c}(3)]); %, 'OutputView','full')); %, 'FillValues', mean(dataB(:)))); 
    dataB_2 = circshift(dataB_1,[-p{c}(1), -p{c}(2), -p{c}(3)]); %, 'OutputView','full')); %, 'FillValues', mean(dataB(:)))); 
    shiftB=[];
    shiftA=[];
    if p{c}(1)>0 & p{c}(1)<=sy & p{c}(2)>0 & p{c}(2)<=sx & p{c}(3)>0 & p{c}(3)<=sz % > > > a a a
          shiftB = dataB_2(1:end-p{c}(1), 1:end-p{c}(2),1:end-p{c}(3));
          shiftA = dataA(1:end-p{c}(1), 1:end-p{c}(2),1:end-p{c}(3));      
    elseif p{c}(1)<=0 & p{c}(1)>-sy & p{c}(2)<=0 & p{c}(2)>-sx & p{c}(3)<=0 & p{c}(3)>-sz % < < < b b b         
          shiftB = dataB_2(-p{c}(1)+1:end,-p{c}(2)+1:end,-p{c}(3)+1:end);
          shiftA = dataA(-p{c}(1)+1:end,-p{c}(2)+1:end,-p{c}(3)+1:end);  
    elseif p{c}(1)>0 & p{c}(1)<=sy & p{c}(2)<=0 & p{c}(2)>-sx & p{c}(3)<=0 & p{c}(3)>-sz % > < < a b b
        shiftB = dataB_2(1:end-p{c}(1),-p{c}(2)+1:end, -p{c}(3)+1:end);
        shiftA = dataA(1:end-p{c}(1),-p{c}(2)+1:end, -p{c}(3)+1:end);
    elseif p{c}(1)<=0 & p{c}(1)>-sy & p{c}(2)>0 & p{c}(2)<=sx & p{c}(3)>0 & p{c}(3)<=sz % < > > b a a
        shiftB = dataB_2(-p{c}(1)+1:end,1:end-p{c}(2),1:end-p{c}(3));
        shiftA = dataA(-p{c}(1)+1:end,1:end-p{c}(2),1:end-p{c}(3));
     elseif p{c}(1)>0 & p{c}(1)<=sy & p{c}(2)>0 & p{c}(2)<=sx & p{c}(3)<=0 & p{c}(3)>-sz % > > < a a b
         shiftB = dataB_2(1:end-p{c}(1),1:end-p{c}(2),-p{c}(3)+1:end);
         shiftA = dataA(1:end-p{c}(1),1:end-p{c}(2),-p{c}(3)+1:end);
    elseif p{c}(1)<=0 & p{c}(1)>-sy & p{c}(2)>0 & p{c}(2)<=sx & p{c}(3)<=0 & p{c}(3)>-sz% < > < b a b
        shiftB = dataB_2(-p{c}(1)+1:end,1:end-p{c}(2),-p{c}(3)+1:end);
        shiftA = dataA(-p{c}(1)+1:end,1:end-p{c}(2),-p{c}(3)+1:end);
    elseif p{c}(1)>0 & p{c}(1)<=sy & p{c}(2)<=0 & p{c}(2)>-sx & p{c}(3)>0 & p{c}(3)<=sz  % > < > a b a
        shiftB = dataB_2(1:end-p{c}(1),-p{c}(2)+1:end,1:end-p{c}(3));
        shiftA = dataA(1:end-p{c}(1),-p{c}(2)+1:end,1:end-p{c}(3));
    elseif p{c}(1)<=0 & p{c}(1)>sy & p{c}(2)<=0 & p{c}(2)>-sx & p{c}(3)>0 & p{c}(3)<=sz % < < > b b a
        shiftB = dataB_2(-p{c}(1)+1:end, -p{c}(2)+1:end, 1:end-p{c}(3));
        shiftA = dataA(-p{c}(1)+1:end, -p{c}(2)+1:end, 1:end-p{c}(3));
    end

    shiftA = shiftA - mean(shiftA(:));
    shiftB = shiftB - mean(shiftB(:));
    Coef = shiftA.*shiftB/(std(shiftA(:))*std(shiftB(:)));
    [ny, nx, nz] = size(shiftA);
    r(c) = sum(Coef(:))/(ny*nx*nz);
    
    end
end

r(isnan(r))=0;
r(isinf(r))=0;

r

index = find(r==max(r));
index = index(1);

if SubPixelFlag == 1
    a = CoarseTraslation - [Py2, Px2, Pz2];
else
    a = CoarseTraslation;
end
  
b = CoarseTraslation - [sy, sx, sz];
p{1} = [a(1), a(2), a(3)];
p{2} = [a(1), a(2), b(3)];
p{3} = [a(1), b(2), a(3)];
p{4} = [a(1), b(2), b(3)];
p{5} = [b(1), a(2), a(3)];
p{6} = [b(1), a(2), b(3)];
p{7} = [b(1), b(2), a(3)];
p{8} = [b(1), b(2), b(3)];

RegB = imtranslate(dataB_1, [-p{index}(2), -p{index}(1), -p{index}(3)]); %, 'OutputView','full')); %, 'FillValues', mean(dataB(:)))); 

if GPU_Flag == 1
    ce = gather(r(index));  % correlation coefficient
else
    ce = r(index);
end

Matrix = [p{index}(1), p{index}(2), p{index}(3)]; % y, x, z
y_shift = (p{index}(1) + sy)/sy;
x_shift = (p{index}(2) + sx)/sx;
z_shift = (p{index}(3) + sz)/sz;
Overlap = [y_shift, x_shift, z_shift];  %overlap between the adjecnt tiles
% NCC(RegB,dataA)

 end