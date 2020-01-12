function r = NCC(dataA, dataB)    

    dataA = single(dataA);
    dataB = single(dataB);
    dataA = dataA - mean(dataA(:));
    dataB = dataB - mean(dataB(:));
    Coef = dataA.*dataB/(std(dataA(:))*std(dataB(:)));
    [ny, nx, nz] = size(dataA);
    r = sum(Coef(:))/(ny*nx*nz);