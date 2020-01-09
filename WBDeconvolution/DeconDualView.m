% DualViewDcon.m: dual-view joint RL deconvolution for time-lapse images, compatible
% with unmatched back projectors
clear all;

% load raw data
[filename_data, path_data] = uigetfile('*.tif','Choose any one of raw data');
% load PSF data 
[filename_psf, path_psf] = uigetfile('*.tif','Choose any one of PSF image',path_data);

% set parameters
dlg_title = 'Set Parameters';
prompt = {'Enter deconvolution method: 1 for traditional decon; 2 for WB',...
    'Enter processing mode: 0 for CPU; 1 for GPU', 'Enter iteration number: ', ...
    'Enter time points to be processed'};
num_lines = 2;
defaultans = {'2','1','1','0-2'};
answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
deconMethod = str2num(answer{1});
proMode = str2num(answer{2});
itNum = str2num(answer{3}); % iteration number
timepoints = strsplit(answer{4},'-');
t1 = str2num(timepoints{1});
t2 = str2num(timepoints{2});
gpuFlag = 0;
if(proMode==1)
    gpuFlag = 1;% 0: CPU; 1: GPU  
    gpuDevNum = 1; % specify the GPU device if there are multiple GPUs
end
path_output = [path_data, 'results\'];
mkdir(path_output);
%%%%%%%%%%%%%%%%%%%%%%%% read in images %%%%%%%%%%%%%%%%%%%%%
stackIn = single(ReadTifStack([path_data, filename_data]));
[Sx, Sy, Sz] = size(stackIn);

disp('Preprocessing forward and back projectors ...');
% % forward projectors: PSFA and PSFB
PSFIn = single (ReadTifStack([path_psf, 'PSFA.tif']));
PSF1 = PSFIn/sum(PSFIn(:));
PSFIn = single (ReadTifStack([path_psf, 'PSFB.tif']));
PSF2 = PSFIn/sum(PSFIn(:));
% % % back projector: PSF_bp
% parameters: light sheet microscopy as an example
switch(deconMethod)
    case 1
        PSF3 = flipPSF(PSF1);
        PSF4 = flipPSF(PSF2);
    case 2
        PSFIn = single (ReadTifStack([path_psf, 'PSFA_BP.tif']));
        PSF3 = PSFIn/sum(PSFIn(:));
        PSFIn = single (ReadTifStack([path_psf, 'PSFB_BP.tif']));
        PSF4 = PSFIn/sum(PSFIn(:));
    otherwise
        error('Processing terminated, please set deconconvolution method as 1 or 2')
end

% % % deconvolution
PSFA_fp = align_size(PSF1, Sx,Sy,Sz);
PSFB_fp = align_size(PSF2, Sx,Sy,Sz);
PSFA_bp = align_size(PSF3, Sx,Sy,Sz);
PSFB_bp = align_size(PSF4, Sx,Sy,Sz);
if(gpuFlag)
    g = gpuDevice(gpuDevNum); 
    reset(g); wait(g); 
    disp(['GPU free memory: ', num2str(g.FreeMemory/1024/1024), ' MB']);
end
if(gpuFlag)
    OTFA_fp = fftn(ifftshift(gpuArray(single(PSFA_fp))));
    OTFB_fp = fftn(ifftshift(gpuArray(single(PSFB_fp))));
    OTFA_bp = fftn(ifftshift(gpuArray(single(PSFA_bp))));
    OTFB_bp = fftn(ifftshift(gpuArray(single(PSFB_bp))));
else
    OTFA_fp = fftn(ifftshift(PSFA_fp));
    OTFB_fp = fftn(ifftshift(PSFB_fp));
    OTFA_bp = fftn(ifftshift(PSFA_bp));
    OTFB_bp = fftn(ifftshift(PSFB_bp));
end
disp('Start deconvolution...');
smallValue = 0.01;
for imgNum = t1:t2
    disp(['...Processing image #: ' num2str(imgNum)]);
    fileInA = [path_data, 'StackA_', num2str(imgNum), '.tif']; %
    fileInB = [path_data, 'StackB_', num2str(imgNum), '.tif']; %
    stackInA = single(ReadTifStack(fileInA));
    stackInB = single(ReadTifStack(fileInB));
    if(gpuFlag)
        stackA = gpuArray(single(stackInA));
        stackB = gpuArray(single(stackInB));
    else
        stackA = stackInA;
        stackB = stackInB;
    end
    stackA = max(stackA,smallValue);
    stackB = max(stackB,smallValue);
    stackEstimate = (stackA + stackB)/2;
    for i = 1:itNum         
        stackEstimate = stackEstimate.*ConvFFT3_S(stackA./...
        ConvFFT3_S(stackEstimate, OTFA_fp),OTFA_bp);
        stackEstimate = max(stackEstimate,smallValue);
        
        stackEstimate = stackEstimate.*ConvFFT3_S(stackB./...
        ConvFFT3_S(stackEstimate, OTFB_fp),OTFB_bp);
        stackEstimate = max(stackEstimate,smallValue);
    end
    if(gpuFlag)
        output = gather(stackEstimate);
    else
        output = stackEstimate;
    end
    WriteTifStack(output, [path_output, 'Decon_', num2str(imgNum), '.tif'], 32);
end
if(gpuFlag) % reset GPU
    reset(g); 
end
disp('Deconvolution completed !!!');

%%%%%%%%%%%%%%%%%%%%%%%%
% % % Functions
function img2 = align_size(img1,Sx2,Sy2,Sz2,padValue)
if(nargin == 4)
    padValue = 0;
end

[Sx1,Sy1,Sz1] = size(img1);
Sx = max(Sx1,Sx2);
Sy = max(Sy1,Sy2);
Sz = max(Sz1,Sz2);
imgTemp = ones(Sx,Sy,Sz)*padValue;

Sox = round((Sx-Sx1)/2)+1;
Soy = round((Sy-Sy1)/2)+1;
Soz = round((Sz-Sz1)/2)+1;
imgTemp(Sox:Sox+Sx1-1,Soy:Soy+Sy1-1,Soz:Soz+Sz1-1) = img1;


Sox = round((Sx-Sx2)/2)+1;
Soy = round((Sy-Sy2)/2)+1;
Soz = round((Sz-Sz2)/2)+1;
img2 = imgTemp(Sox:Sox+Sx2-1,Soy:Soy+Sy2-1,Soz:Soz+Sz2-1);
end

function outPSF = flipPSF(inPSF)
% function outPSF = flipPSF(inPSF)
% outPSF(i,j,k) = inPSF(m-i+1,n-j+1,l-k+1);
[Sx, Sy, Sz] = size(inPSF);
outPSF = zeros(Sx, Sy, Sz);
for i = 1:Sx
    for j = 1:Sy
        for k = 1:Sz
            outPSF(i,j,k) = inPSF(Sx-i+1,Sy-j+1,Sz-k+1);
        end
    end
end
end