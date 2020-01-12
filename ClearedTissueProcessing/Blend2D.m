function dataC = Blend(dataA, dataB, dim, overlap, option)

if isempty(dataA)
    dataC = dataB;
elseif dim == 2
       ny = size(dataA,1);
       temp_start = dataA(:, end-overlap+1: end);
       temp_end = dataB(:, 1:overlap);
    if option == 1   % simple average   
       temp = (temp_start + temp_end)/2;       
    elseif option == 2   % simple replacement 
       temp = [temp_start(:,1:overlap/2), temp_start(:, overlap/2+1:end)];
    elseif option == 3  % linear blend
       ramp = linspace(0, 1, overlap);
       ramp_start = repmat(ramp, [ny, 1]);
       temp = temp_start.*(1- ramp_start) + temp_end.* ramp_start;    
    end
       
     dataC = [dataA(:, 1: end-overlap), temp, dataB(:, overlap+1:end)];
     
  elseif dim == 1
   
       nx = size(dataA,2);
       temp_start = dataA(end-overlap+1:end,:);
       temp_end = dataB(1:overlap,:);
    if option == 1   % simple average   
       temp = (temp_start + temp_end)/2;       
    elseif option == 2   % simple replacement 
       temp = [temp_start(1:overlap/2,:), temp_start(overlap/2+1:end,:)];
    elseif option == 3  % linear blend
       ramp = linspace(0, 1, overlap)';
       ramp_start = repmat(ramp, [1, nx]);     
       temp = temp_start.*(1- ramp_start) + temp_end.* ramp_start;    
    end
       
     dataC = [dataA(1: end-overlap,:); temp; dataB(overlap+1:end,:)];
     
end
end




