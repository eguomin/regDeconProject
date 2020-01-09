% SingleViewDcon.m: single view RL deconvolution for time-lapse images, compatible
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
% % forward projector: PSF
PSFIn = single (ReadTifStack([path_psf, filename_psf]));
PSF1 = PSFIn/sum(PSFIn(:));
% % % back projector: PSF_bp
% parameters: light sheet microscopy as an example
switch(deconMethod)
    case 1
        bp_type = 'traditional'; 
    case 2
        bp_type = 'wiener-butterworth';
    otherwise
        error('Processing terminated, please set deconconvolution method as 1 or 2')
end
alpha = 0.05;
beta = 1; 
n = 10;
resFlag = 1;
iRes = [2.44,2.44,10];
verboseFlag = 0;
[PSF2, ~] = BackProjector(PSF1, bp_type, alpha, beta, n, resFlag, iRes, verboseFlag);
PSF2 = PSF2/sum(PSF2(:));
WriteTifStack(PSF1, [path_output, 'PSF_fp.tif'], 32);
WriteTifStack(PSF2, [path_output, 'PSF_bp.tif'], 32);

% set initialization of the deconvolution
flagConstInitial = 0; % 1: constant mean; 0: input image

% % % deconvolution
PSF_fp = align_size(PSF1, Sx,Sy,Sz);
PSF_bp = align_size(PSF2, Sx,Sy,Sz);
if(gpuFlag)
    g = gpuDevice(gpuDevNum); 
    reset(g); wait(g); 
    disp(['GPU free memory: ', num2str(g.FreeMemory/1024/1024), ' MB']);
end
if(gpuFlag)
    OTF_fp = fftn(ifftshift(gpuArray(single(PSF_fp))));
    OTF_bp = fftn(ifftshift(gpuArray(single(PSF_bp))));
else
    OTF_fp = fftn(ifftshift(PSF_fp));
    OTF_bp = fftn(ifftshift(PSF_bp));
end
disp('Start deconvolution...');
smallValue = 0.001;
for imgNum = t1:t2
    disp(['...Processing image #: ' num2str(imgNum)]);
    fileIn = [path_data, 'Stack_', num2str(imgNum), '.tif']; %
    stackIn = single(ReadTifStack(fileIn));
    if(gpuFlag)
        stack = gpuArray(single(stackIn));
    else
        stack = stackIn;
    end
    stack = max(stack,smallValue);
    if(flagConstInitial==1)
        stackEstimate = ones(Sx, Sy, Sz)*mean(stack(:)); % constant initialization
    else
        stackEstimate = stack; % Measured image as initialization
    end
    
    for i = 1:itNum
        stackEstimate = stackEstimate.*ConvFFT3_S(stack./...
        ConvFFT3_S(stackEstimate, OTF_fp),OTF_bp);
        stackEstimate = max(stackEstimate,smallValue);
    end
    if(gpuFlag)
        output = gather(stackEstimate);
    else
        output = stackEstimate;
    end
    WriteTifStack(output, [path_output, 'Decon_', num2str(imgNum), '.tif'], 16);
end
if(gpuFlag) % reset GPU
    reset(g); 
end
disp('Deconvolution completed !!!');

%%%%%%%%%%%%%%%%%%%%%%%%
% % % Function
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