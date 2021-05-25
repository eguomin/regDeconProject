function dataC = BlendZ(dataA, dataB, ramp_start, overlap)

if isempty(dataA)
    dataC = dataB;
else
    dataC = dataA.*(1- ramp_start) + dataB.* ramp_start;   
end

end

