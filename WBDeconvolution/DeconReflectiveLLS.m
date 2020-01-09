% DeconReflectiveLLS.m: spatial variant RL deconvolution for reflective 
% lattice light-sheet microscopy, compatible with unmatched back projectors
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
defaultans = {'2','1','5','0-9'};
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
t_start0 = clock;
path_output = [path_data, 'results\'];
mkdir(path_output);
%%%%%%%%%%%%%%%%%%%%%%%% read in images %%%%%%%%%%%%%%%%%%%%%
stackIn = single(ReadTifStack([path_data, filename_data]));
[Sx, Sy, Sz] = size(stackIn);
N = size(stackIn);

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

% % % PSFs
PSFA_fp = align_size(PSF1, Sx,Sy,Sz);
PSFB_fp = align_size(PSF2, Sx,Sy,Sz);
PSFA_bp = align_size(PSF3, Sx,Sy,Sz);
PSFB_bp = align_size(PSF4, Sx,Sy,Sz);

% set parameters
pixel = 0.105;
angle = 31;
% interval number 
ny = N(1);
nx_img = N(2);
nz_img = N(3);

N_Image = [ny, nx_img, nz_img];  %size of image
nx_obj = nx_img;          
nz_obj = nz_img;
N_Obj = [ny, nx_obj, nz_obj];  % size of object
N_PSF = N_Image;               % size of PSF
N_Exc = [ny, 2*nx_obj, nz_obj];  % size of Excitation pattern

smallValue = 0.001;
% GPU setup
if(gpuFlag)
    g = gpuDevice(gpuDevNum); 
    reset(g); wait(g); 
    disp(['GPU free memory: ', num2str(g.FreeMemory/1024/1024), ' MB']);
end
if(gpuFlag)
    OTF_A = fftn(ifftshift(gpuArray(single(PSFA_fp))));
    OTF_B = fftn(ifftshift(gpuArray(single(PSFB_fp))));
    OTF_C = fftn(ifftshift(gpuArray(single(PSFA_bp))));
    OTF_D = fftn(ifftshift(gpuArray(single(PSFB_bp))));
else
    OTF_A = fftn(ifftshift(single(PSFA_fp)));
    OTF_B = fftn(ifftshift(single(PSFB_fp)));
    OTF_C = fftn(ifftshift(single(PSFA_bp)));
    OTF_D = fftn(ifftshift(single(PSFB_bp)));
end
WriteTifStack(PSFA_fp, [path_output, 'PSFA_fp.tif'], 32);
WriteTifStack(PSFB_fp, [path_output, 'PSFB_fp.tif'], 32);
WriteTifStack(PSFA_bp, [path_output, 'PSFA_bp.tif'], 32);
WriteTifStack(PSFB_bp, [path_output, 'PSFB_bp.tif'], 32);
if(gpuFlag)
    disp(['OTFs Done! GPU free memory: ', num2str(g.FreeMemory/1024/1024), ' MB']);
else
    disp('OTFs Done!');
end

% excitation pattern
UnityLine = zeros(nz_obj,2*nx_obj,'single');
UnityLine(nz_obj/2,:) = 1.0;
UnityLineA = imrotate(UnityLine,angle,'bilinear','crop');
LineA = permute(repmat(UnityLineA,[1, 1, ny]),[3,2,1]);
LineB = flip(LineA,3);
if(gpuFlag)
    LineA = gpuArray(single(LineA));
    LineB = gpuArray(single(LineB));
else
    LineA = single(LineA);
    LineB = single(LineB);
end

% excitation pattern
Excitation = single(ReadTifStack([path_psf,'Excitation.tif']));
S_Exc = size(Excitation);
Sox = round((S_Exc(1)-N_Exc(1))/2) + 1;
Soy = round((S_Exc(2)-N_Exc(2))/2) + 1;
Excitation = Excitation(Sox:Sox+N_Exc(1)-1,Soy:Soy+N_Exc(2)-1,:);
WriteTifStack(Excitation, [path_output, 'Excitation.tif'], 32);
if(gpuFlag)
    Excitation = gpuArray(single(Excitation));
    disp(['Excitation Done! GPU free memory: ', num2str(g.FreeMemory/1024/1024), ' MB']);
else
    disp('Excitation Done!');
end

% calculationg the sensitivity 
if(gpuFlag)
    SensA = zeros([ny, nx_obj, nz_obj],'single', 'gpuArray'); 
else
    SensA = zeros([ny, nx_obj, nz_obj],'single'); 
end 
for x = 1 : nx_img
        px_range = x: x + nx_img -1;           
        InterMediate = LineA(:,px_range,:); 
        Sample_Illuminate = ConvFFT3_S(InterMediate,OTF_C).* Excitation(:, px_range,:); 
        Sample_Illuminate = max(Sample_Illuminate, 0);
        SensA = SensA + single(Sample_Illuminate);
end

if(gpuFlag)
    SensA = zeros([ny, nx_obj, nz_obj],'single', 'gpuArray'); 
else
    SensA = zeros([ny, nx_obj, nz_obj],'single'); 
end  
for x = 1 : nx_img
        px_range = x: x + nx_img -1;           
        InterMediate = LineB(:,px_range,:); 
        Sample_Illuminate = ConvFFT3_S(InterMediate,OTF_D).* Excitation(:, px_range,:); 
        Sample_Illuminate = max(Sample_Illuminate, 0);
        SensB = SensB + single(Sample_Illuminate);
end
SensA = max(SensA, smallValue);
SensB = max(SensB, smallValue);

%calculationg the sensitivity of A 
if(gpuFlag)
    SensA = zeros([ny, nx_obj, nz_obj],'single', 'gpuArray'); 
else
    SensA = zeros([ny, nx_obj, nz_obj],'single'); 
end
if(gpuFlag)
    WriteTifStack(gather(SensA),  [path_output, 'SensA.tif'], 32);
    WriteTifStack(gather(SensB),  [path_output, 'SensB.tif'], 32);
    disp(['Sensitivity Done! GPU free memory: ', num2str(g.FreeMemory/1024/1024), ' MB']);
else
    WriteTifStack(SensA,  [path_output, 'SensA.tif'], 32);
    WriteTifStack(SensB,  [path_output, 'SensB.tif'], 32);
    disp('Sensitivity Done!');
end

disp('Start deconvolution...');
for time= t1:t2
disp(['...Processing image #: ' num2str(time)]);

ViewA = single(ReadTifStack(strcat(path_data,'StackA_',num2str(time),'.tif'))); 
ViewB = single(ReadTifStack(strcat(path_data,'StackB_',num2str(time),'.tif'))); 

if(gpuFlag)
    ViewA = gpuArray(ViewA);
    ViewB = gpuArray(ViewB);
end
ViewA = max(ViewA, smallValue);
ViewB = max(ViewB, smallValue);

Estimate = (ViewA + ViewB)/2;

if(gpuFlag)
    disp(['...GPU Memory: ',num2str(g.FreeMemory)]);
end

for iteration = 1:itNum
   disp(['......Processing iteration # ' num2str(iteration)]);
   t_start = clock;
    
   % View A Forward and Adjoint Model
   if(gpuFlag)
        CorrectionA = zeros([ny, nx_obj, nz_obj],'single', 'gpuArray'); 
   else
        CorrectionA = zeros([ny, nx_obj, nz_obj],'single');   
   end
   for x = 1 : nx_img      
        px_range = x: x + nx_img -1;
        SampleBlur = ConvFFT3_S(Estimate.*Excitation(:,px_range, :), OTF_A); 
        InterMediate = ViewA./SampleBlur.*LineA(:,px_range,:);
        Sample_Illuminate = ConvFFT3_S(InterMediate,OTF_C).*Excitation(:,px_range,:); 
        Sample_Illuminate = max(Sample_Illuminate, 0);
        CorrectionA = CorrectionA + Sample_Illuminate;    
   end    
    Estimate = Estimate.*CorrectionA./SensA; 
    Estimate = max(Estimate, smallValue);
     
    disp(['......View A Model Done!  GPU Memory: ',num2str(g.FreeMemory)]);
    
    % View B Forward and Adjoint Model
    if(gpuFlag)
        CorrectionB = zeros([ny, nx_obj, nz_obj],'single', 'gpuArray'); 
   else
        CorrectionB = zeros([ny, nx_obj, nz_obj],'single');   
   end
   for x = 1 : nx_img      
        px_range = x: x + nx_img -1;
        SampleBlur = ConvFFT3_S(Estimate.*Excitation(:,px_range, :), OTF_B);  
        InterMediate = ViewB./SampleBlur.*gpu_LineB(:,px_range,:);
        Sample_Illuminate = ConvFFT3_S(InterMediate,OTF_D).*Excitation(:,px_range,:); 
        Sample_Illuminate = max(Sample_Illuminate, 0);
        CorrectionB = CorrectionB + Sample_Illuminate; 
    end
        
    Estimate = Estimate.*CorrectionB./SensB; 
    Estimate = max(Estimate, smallValue);
    
    disp(['......View B Model Done! GPU Memory: ',num2str(g.FreeMemory)]);
    disp(['......Iteration # ' num2str(iteration), ' takes ', num2str(etime(clock,t_start)), ' s']);      
end
if(gpuFlag)
    output = gather(Estimate);
else
    output = Estimate;
end
WriteTifStack(output, [path_output, 'Decon_', num2str(imgNum), '.tif'], 32);
end
if(gpuFlag)
    reset(g); 
end
disp('Deconvolution completed !!!');
disp(['Total time cost: ', num2str(etime(clock,t_start0)), ' s']); 
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