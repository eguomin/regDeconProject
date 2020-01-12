function [outVol] = ConvFFT3_S(inVol,OTF)

    outVol = single(real(ifftn(fftn(inVol).*OTF)));  

end
 



