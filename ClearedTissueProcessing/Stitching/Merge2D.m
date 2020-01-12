function dataC = Merge2D(dataA, dataB, dim)

[ny, nx] = size(dataA)

if dim == 2
       ramp = linspace(0, 1, nx);
       ramp_start = repmat(ramp, [ny, 1]);
       dataC = dataA.*(1- ramp_start) + dataB.* ramp_start; 
elseif dim == 1
       ramp = linspace(0, 1, ny)';
       ramp_start = repmat(ramp, [1, nx]);     
       dataC = dataA.*(1- ramp_start) + dataB.* ramp_start;          
     
end




