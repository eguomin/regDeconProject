% % % Generate back projector
clear all;
% % Forward projectors
file_PSF_fp = 'F:\Program\Decon\Wiener-Butterworth Deconvolution in Matlab\DataForTest_lightsheet\PSF.tif';
pathOut = 'F:\Program\Decon\Wiener-Butterworth Deconvolution in Matlab\DataForTest_lightsheet\';
PSF_fp = ReadTifStack(file_PSF_fp);
[Sx, Sy, Sz] = size(PSF_fp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isequal(exist(pathOut, 'dir'),7)
    disp(['output folder:' pathOut]);
else
    mkdir(pathOut);
    disp(['output folder created:' pathOut]);
end
%%%%%%% ********** Parameters *************** %%%%%%
bp_type = 'wiener-butterworth';
alpha = 0.05;
beta = 1; 
n = 10;
resFlag = 2;
% iRes = zeros(1,3);
iRes = [2.44,2.44,10];
verboseFlag = 1;
% % call function: BackProjector
[PSF_bp, OTF_bp] = BackProjector(PSF_fp, bp_type, alpha, beta, n, resFlag, iRes, verboseFlag);
% % Save back projectors
% WriteTifStack(PSF_bp, [pathOut, 'PSF_bp_' bp_type '.tif'], 32);
% OTF_bp = fftshift(abs(OTF_bp))/abs(OTF_bp(1));
% WriteTifStack(OTF_bp, [pathOut, 'OTF_bp_' bp_type '.tif'], 32);



