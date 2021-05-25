clear all;

w = warning('off', 'MATLAB:imagesci:tiffmexutils:libtiffWarning');
warning('off', 'MATLAB:imagesci:tifftagsread:expectedTagDataFormat');

file_path = uigetdir();
file_path = [file_path, '\'];
%file_path = 'Y:\Yicong\Clearing\Gut_FourColor\Test\';

% parameter setting:
DownsamplingFactor = 5; 
RegFlag = 1;  % if do not save reg files, then set RegFlag = 0;
DeconFlag = 1; % if do not run decon, then set DeconFlag = 0;
FlipFlag = 0; % if the input is XZ view
CropStepXY = 512;%512; % crop step size, every 512 pixels 
CropStepZ = 512;%512; % crop step size, every 512 pixels 

% path, file name setting
lib_path = '..\cudaLib\';
lib_name = 'libapi';
filenameA_down = 'AfterStitchSheetModeADownsampling';
filenameB_down = 'AfterStitchSheetModeBDownsampling';
% filenameA = 'StackA_LightsheetMode';
% filenameB = 'StackB_LightsheetMode';
filenameA = 'AfterStitchSheetModeA';
filenameB = 'AfterStitchSheetModeB';

% if users has multiple GPUS, can reset GPUDevice to 1
GPUDevice =2;
if GPUDevice == 1
    GPU_M = 2; GPU_C = 1;
elseif GPUDevice == 2
        GPU_M = 1;GPU_C = 0;
end

% create folders
decon_file_path = [file_path, 'Decon'];
if ~exist(decon_file_path, 'dir')
  mkdir(decon_file_path);
end

reg_file_path = [file_path, 'RegA'];
if ~exist(reg_file_path, 'dir')
  mkdir(reg_file_path);
end

reg_file_path = [file_path, 'RegB'];
if ~exist(reg_file_path, 'dir')
  mkdir(reg_file_path);
end

Merge_file_path = [file_path, 'DeconMergeZ'];
if ~exist(Merge_file_path, 'dir')
  mkdir(Merge_file_path);
end

Merge_file_path = [file_path, 'RegAMergeZ'];
if ~exist(Merge_file_path, 'dir')
  mkdir(Merge_file_path);
end

Merge_file_path = [file_path, 'RegBMergeZ'];
if ~exist(Merge_file_path, 'dir')
  mkdir(Merge_file_path);
end

Merge_file_path = [file_path, 'RegAMergeXY'];
if ~exist(Merge_file_path, 'dir')
  mkdir(Merge_file_path);
end

Merge_file_path = [file_path, 'RegBMergeXY'];
if ~exist(Merge_file_path, 'dir')
  mkdir(Merge_file_path);
end

Downsamplinge_path = [file_path, '\DeconMergeZ\Downsampling\'];
if ~exist(Downsamplinge_path, 'dir')
  mkdir(Downsamplinge_path);
end

Downsamplinge_path = [file_path, '\DeconMergeXY\Downsampling\'];
if ~exist(Downsamplinge_path, 'dir')
  mkdir(Downsamplinge_path);
end


Downsamplinge_path = [file_path, '\RegAMergeXY\Downsampling\'];
if ~exist(Downsamplinge_path, 'dir')
  mkdir(Downsamplinge_path);
end

Downsamplinge_path = [file_path, '\RegBMergeXY\Downsampling\'];
if ~exist(Downsamplinge_path, 'dir')
  mkdir(Downsamplinge_path);
end

Downsamplinge_path = [file_path, '\RegAMergeZ\Downsampling\'];
if ~exist(Downsamplinge_path, 'dir')
  mkdir(Downsamplinge_path);
end

Downsamplinge_path = [file_path, '\RegBMergeZ\Downsampling\'];
if ~exist(Downsamplinge_path, 'dir')
  mkdir(Downsamplinge_path);
end



% Register downsampled files
% Registration(lib_path, lib_name, file_path, filenameA_down, filenameB_down,GPU_C);
disp('Coarse Registration Done !!!'); 

Tmx = dlmread([file_path, filenameB_down, 'RegMat.txt']); %,'delimiter',' ');
Tmx = Tmx';

% switch x, y axis - something different from the dll and Matlab.
Transformation_matrix =[Tmx(2,2), Tmx(2,1), Tmx(2,3), 0; 
        Tmx(1,2), Tmx(1,1), Tmx(1,3), 0;
        Tmx(3,2), Tmx(3,1), Tmx(3,3), 0;
        Tmx(4,2), Tmx(4,1), Tmx(4,3), 1];
Transformation_matrix(4,:) = Transformation_matrix(4,:) * DownsamplingFactor; 
Transformation_matrix(4,4) = 1;
Transformation_matrix = Transformation_matrix';
 
start_t = cputime; 

dataA = TIFFStack([file_path, filenameA, '.tif']);
dataB = TIFFStack([file_path, filenameB, '.tif']);
[nyA, nxA, nzA] = size(dataA);
[nyB, nxB, nzB] = size(dataB);

tilex = ceil(nxA/CropStepXY)
tiley = ceil(nyA/CropStepXY)
tilez = ceil(nzA/CropStepZ)

[nyA, nxA, nzA] = size(dataA);
[nyB, nxB, nzB] = size(dataB);

end_t = cputime - start_t;

disp(['crop registraiton ......']);
CropRegDecon(lib_path, lib_name, file_path, filenameA, filenameB, Transformation_matrix, CropStepXY, CropStepZ, tilex, tiley,tilez,  RegFlag, DeconFlag, FlipFlag, GPU_M, GPU_C);

 disp(['crop registraiton done ! Stithcing now......']); 
if tilez > 1
    if RegFlag == 1
        file_path_in = [file_path '\RegA'];
        file_path_out = [file_path '\RegAMergeZ'];
        StitchingZ(file_path_in, file_path_out, CropStepXY, CropStepZ, tilex, tiley, tilez);
    
        file_path_in = [file_path '\RegB'];
        file_path_out = [file_path '\RegBMergeZ'];
        StitchingZ(file_path_in, file_path_out, CropStepXY, CropStepZ, tilex, tiley, tilez);
        
        file_path_in = [file_path '\RegAMergeZ'];
        file_path_out = [file_path '\RegAMergeXY'];     
        StitchingXY(file_path_in, file_path_out, CropStepXY, CropStepZ, tilex, tiley, nzA);
        
        file_path_in = [file_path '\RegBMergeZ'];
        file_path_out = [file_path '\RegBMergeXY'];     
        StitchingXY(file_path_in, file_path_out, CropStepXY, CropStepZ, tilex, tiley, nzA);   
    end 
    
     if DeconFlag == 1
        file_path_in = [file_path '\Decon'];
        file_path_out = [file_path '\DeconMergeZ'];
        StitchingZ(file_path_in, file_path_out, CropStepXY, CropStepZ, tilex, tiley, tilez);
 
        file_path_in = [file_path '\DeconMergeZ'];
        file_path_out = [file_path '\DeconMergeXY'];     
        StitchingXY(file_path_in, file_path_out, CropStepXY, CropStepZ, tilex, tiley, nzA);   
    end 
        
else 
    
    if RegFlag == 1          
        file_path_in = [file_path '\RegA'];
        file_path_out = [file_path '\RegAMergeXY'];     
        StitchingXY(file_path_in, file_path_out, CropStepXY, CropStepZ, tilex, tiley, nzA);
        
        file_path_in = [file_path '\RegB'];
        file_path_out = [file_path '\RegBMergeXY'];     
        StitchingXY(file_path_in, file_path_out, CropStepXY, CropStepZ, tilex, tiley, nzA);   
    end 
    
     if DeconFlag == 1
        file_path_in = [file_path '\Decon'];
        file_path_out = [file_path '\DeconMergeXY'];     
        StitchingXY(file_path_in, file_path_out, CropStepXY, CropStepZ, tilex, tiley, nzA);   
    end 
end

 disp(['Task Done !!!']); 

function CropRegDecon(lib_path, lib_name, file_path, filenameA, filenameB, Transformation_matrix, CropStepXY, CropStepZ, tilex, tiley, tilez, RegFlag, DeconFlag, FlipFlag, GPU_M, GPU_C) 

dataA = TIFFStack([file_path, filenameA, '.tif']);
dataB = TIFFStack([file_path, filenameB, '.tif']);
% Mask = TIFFStack(filenameMask);
[nyA, nxA, nzA] = size(dataA);
[nyB, nxB, nzB] = size(dataB);
%[nyM, nxM, nzM] = size(Mask)

xSizeF = CropStepXY; ySizeF = CropStepXY; zSizeF = CropStepZ; %final image size for stiching
xSize = xSizeF + 128; ySize = ySizeF + 128; zSize = zSizeF + 128; %initial crop size 
boundary_pixel = 20; % remove decon boundary
overlap_pixel = 128 - boundary_pixel * 2; % overlapp region 

sizeA = [ySize, xSize, zSize];
sizeB = sizeA + 100;

% tilex = ceil(nxA/xSizeF)
% tiley = ceil(nyA/ySizeF)
% tilez = ceil(nzA/zSizeF)

stackA = zeros(sizeA);
stackB = zeros(sizeB);

% % % % % **********create arguments **********
% % % results 
% % % load DLL lib
libFile = [lib_path, lib_name, '.dll'];
libHFile = [lib_path, lib_name, '.h'];
loadlibrary(libFile, libHFile);

 % % check lib functions
mFunctions = libfunctions(lib_name,'-full');
h_regB = libpointer('singlePtr',stackA); % registration feedback pointer: registered B stack

% % % input images and PSFs
tic;
h_tiffA = libpointer('singlePtr',stackA);
h_tiffB = libpointer('singlePtr',stackB);
tifSizeA = libpointer('uint32Ptr',sizeA); % image size pointer
tifSizeB = libpointer('uint32Ptr',sizeB); % image size pointer
cTime1 = toc;

% % %  configurations
Unity_matrix = [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1]; 
regTrigger = 1; % whether or not do registration, 1: yes; 0: no
regMethod = 6; % registration method
                % 1: translation only; 
                % 2: rigid body; 
                % 3: 7 degrees of freedom (translation, rotation, scaling equally in 3 dimemsions)
                % 4: 9 degrees of freedom(translation, rotation, scaling); 
                % 5: 12 degrees of freedom; 
                % 6: rigid body first, then do 12 degrees of freedom
inputTmx = 1; % whether or not use input matrix, 1: yes; 0: no
tmxPtr = libpointer('singlePtr',Unity_matrix');  % input matrix pointer
FTOL = 0.0001; % reg threshold
itLimit = 2000; % maximun iteration number for registration
subBgtrigger = 0; % subtract background (mean value of the image) for registration, 1: yes, 0: no

deviceNum = GPU_C; % GPU device
regRecords = zeros(1,11); % records and feedback of registration
regRecordsPtr = libpointer('singlePtr',regRecords); 

if DeconFlag == 1
    RawPSFA = TIFFStack([file_path '\PSF\PSFA.tif']);
    RawPSFB = TIFFStack([file_path '\PSF\PSFB.tif']);   
    RawPSFC = TIFFStack([file_path '\PSF\PSFA_bp_wiener-butterworth.tif']);
    RawPSFD = TIFFStack([file_path '\PSF\PSFB_bp_wiener-butterworth.tif']);
    [nyP,nxP,nzP] = size(RawPSFA);
    xOff = floor((-nxP + xSize)/2) + 1;     
    yOff = floor((-nyP + ySize)/2) + 1;
    zOff = floor((-nzP + zSize)/2) + 1; 
    PSFA = zeros(sizeA); % + 0.001;
    PSFB = zeros(sizeA); %+ 0.001;
    PSFC = zeros(sizeA); %+ 0.001;
    PSFD = zeros(sizeA); %+ 0.001;
    PSFA(yOff:end-yOff+1,xOff:end-xOff+1,zOff:end-zOff+1) = RawPSFA;
    PSFB(yOff:end-yOff+1,xOff:end-xOff+1,zOff:end-zOff+1) = RawPSFB;
    PSFC(yOff:end-yOff+1,xOff:end-xOff+1,zOff:end-zOff+1) = RawPSFC;
    PSFD(yOff:end-yOff+1,xOff:end-xOff+1,zOff:end-zOff+1) = RawPSFD;

    PSFA = PSFA/sum(PSFA(:));
    PSFB = PSFB/sum(PSFB(:));
    PSFC = PSFC/sum(PSFC(:));
    PSFD = PSFD/sum(PSFD(:));

    g = gpuDevice(GPU_M); reset(g); wait(g);
    OTF_A = gather(fftn(circshift(single(gpuArray(PSFA)),-floor(sizeA/2))));
    OTF_B = gather(fftn(circshift(single(gpuArray(PSFB)),-floor(sizeA/2))));
    OTF_C = gather(fftn(circshift(single(gpuArray(PSFC)),-floor(sizeA/2))));
    OTF_D = gather(fftn(circshift(single(gpuArray(PSFD)),-floor(sizeA/2))));
    reset(g);
end

tile_array = 0:tilez*tiley*tilex-1;

for tile_number = 1:size(tile_array,2)
    tile = tile_array(tile_number)
    z = floor(tile/(tiley*tilex));
    xy = mod(tile, tiley*tilex);  
    x = floor(xy/tiley);
    y = mod(xy, tiley); 
    tic;
    
    tile = z*tiley*tilex + x*tiley + y 
    
    disp(['start reading file:']); 
    
     dataA = TIFFStack([file_path, filenameA, '.tif']);
    dataB = TIFFStack([file_path, filenameB, '.tif']);
 
    start_t = cputime; 
    x_start = 1 + xSizeF * x; 
    y_start = 1 + ySizeF * y; 
    z_start = 1 + zSizeF * z;

    A_start = [x_start, y_start, z_start, 1];
    A_end = [x_start+sizeA(2)-1, y_start+sizeA(1)-1, z_start+sizeA(3)-1, 1]; 
    B_start = round(A_start * Transformation_matrix');
    B_end = [B_start(1)+sizeB(2)-1, B_start(2)+sizeB(1)-1, B_start(3)+sizeB(3)-1, 1];

    Translation_A = Unity_matrix; 
    Translation_A (1:3,4) =  A_start(1:3) - 1;
    Translation_B = Unity_matrix; 
    Translation_B (1:3,4) =  B_start(1:3) - 1;
    New_Transformation_matrix = inv(Translation_B) * Transformation_matrix * Translation_A;
    Tmx_Matlab = inv(New_Transformation_matrix)';
    tform = affine3d(Tmx_Matlab);
    Tmx = New_Transformation_matrix';
    Tmx_C =[Tmx(2,2), Tmx(2,1), Tmx(2,3), 0; 
        Tmx(1,2), Tmx(1,1), Tmx(1,3), 0;
        Tmx(3,2), Tmx(3,1), Tmx(3,3), 0;
        Tmx(4,2), Tmx(4,1), Tmx(4,3), 1];

    tmxPtr.Value = Tmx_C;

    A_i_range = max(A_start(1),1): min(A_end(1), nxA); 
    A_j_range = max(A_start(2),1): min(A_end(2), nyA); 
    A_k_range = max(A_start(3),1): min(A_end(3), nzA); 

    B_i_range = max(B_start(1),1): min(B_end(1), nxB);
    B_j_range = max(B_start(2),1): min(B_end(2), nyB);
    B_k_range = max(B_start(3),1): min(B_end(3), nzB);

    % Mask_i_range = max(floor(A_start(1)/DownsamplingFactor),1): min(floor(A_end(1)/DownsamplingFactor), nxM); 
    % Mask_j_range = max(floor(A_start(2)/DownsamplingFactor),1): min(floor(A_end(2)/DownsamplingFactor), nyM); 
    % Mask_k_range = max(floor(A_start(3)/DownsamplingFactor),1): min(floor(A_end(3)/DownsamplingFactor), nzM); 
    % crop_Mask = Mask(Mask_j_range, Mask_i_range, Mask_k_range);
    % Flag = max(crop_Mask(:));
    Flag = 1;
    
    if Flag == 0
     % skip the registration, deconvolution, and writing based on Mask 
        if FlipFlag == 1
            WriteTifStack(stackA(boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel,boundary_pixel+1:end-boundary_pixel), [file_path '\Decon\Decon_TileX_' num2str(x) '_TileY_' num2str(z) '_TileZ_' num2str(y) '.tif'], '16');
        else 
            WriteTifStack(stackA(boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel,boundary_pixel+1:end-boundary_pixel), [file_path '\Decon\Decon_TileX_' num2str(x) '_TileY_' num2str(y) '_TileZ_' num2str(z) '.tif'], '16');
        end
    else 
    
    parfor kk=1:length(A_k_range)
        crop_A(:,:,kk) = dataA(A_j_range, A_i_range, A_k_range(kk));
    end

    parfor kk=1:length(B_k_range)
        crop_B(:,:,kk) = dataB(B_j_range, B_i_range, B_k_range(kk));
    end

    crop_A(sizeA(1),sizeA(2),sizeA(3)) = 0;
    crop_B(sizeB(1),sizeB(2),sizeB(3)) = 0;

    cTime1 = toc
    h_tiffA.Value = crop_A;
    h_tiffB.Value = crop_B;

    cTime2 = toc
    cudaStatus = calllib(lib_name,'reg_3dgpu',h_regB, tmxPtr, h_tiffA, h_tiffB,tifSizeA,...
        tifSizeB, regMethod,inputTmx, FTOL, itLimit, subBgtrigger, deviceNum, regRecordsPtr);
    cTime3 = toc
    % % % save images
    crop_B_reg = single(reshape(h_regB.Value,sizeA)) + 0.001;
    dlmwrite([file_path, '\RegA\RegMat_' num2str(tile), '.txt'],tmxPtr.Value)

    if DeconFlag == 1
        crop_A = single(crop_A) + 0.001;
        Estimate = (crop_A + crop_B_reg)/2;
        for iteration = 1:1
            %disp(['Processing iteration # ', num2str(iteration)]);
            gpu_Estimate = gpuArray(Estimate);
            gpu_Temp = abs(ConvFFT3_S(gpu_Estimate, gpuArray(OTF_A)))+ 0.001;
            gpu_Temp = gpuArray(crop_A./gather(gpu_Temp));
            gpu_Temp = abs(ConvFFT3_S(gpu_Temp, gpuArray(OTF_C)))+ 0.001;
            gpu_Estimate = gpu_Estimate.*gpu_Temp;
     
            gpu_Temp = abs(ConvFFT3_S(gpu_Estimate, gpuArray(OTF_B))) + 0.001;
            gpu_Temp = gpuArray(crop_B_reg./gather(gpu_Temp));
            gpu_Temp = abs(ConvFFT3_S(gpu_Temp, gpuArray(OTF_D))) + 0.001;
            gpu_Estimate = gpu_Estimate.*gpu_Temp;
 
            disp(['Decon Done!  GPU Memory: ',num2str(g.FreeMemory/1e9)]);   
        end

        cTim4 = toc;
    end 

%      movefile('log.txt', [file_path, '\RegA\','log' num2str(tile), '.txt']);
     if RegFlag == 1
        if FlipFlag == 1
          crop_A_1 = permute(crop_A(boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel),[3,2,1]);
          crop_B_reg_1 = permute(crop_B_reg(boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel),[3,2,1]);
          WriteTifStack(crop_A_1, [file_path '\RegA\', 'TileX_' num2str(x) '_TileY_' num2str(z) '_TileZ_' num2str(y) '.tif'], '16');
          WriteTifStack(crop_B_reg_1, [file_path '\RegB\', 'TileX_' num2str(x) '_TileY_' num2str(z) '_TileZ_' num2str(y) '.tif'], '16');
        else 
            crop_A_1 = crop_A(boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel);
            crop_B_reg_1 = crop_B_reg(boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel);
            WriteTifStack(crop_A_1, [file_path '\RegA\', 'TileX_' num2str(x) '_TileY_' num2str(y) '_TileZ_' num2str(z) '.tif'], '16');
            WriteTifStack(crop_B_reg_1, [file_path '\RegB\', 'TileX_' num2str(x) '_TileY_' num2str(y) '_TileZ_' num2str(z) '.tif'], '16');
        end
     end
 
    clear crop_A
    clear crop_B

    if DeconFlag == 1
        if FlipFlag == 1
            Decon = gather(permute(gpu_Estimate(boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel),[3,2,1]));
            WriteTifStack(Decon, [file_path '\Decon\', 'TileX_' num2str(x) '_TileY_' num2str(z) '_TileZ_' num2str(y) '.tif'], '16');
        else 
            Decon = gather(gpu_Estimate(boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel, boundary_pixel+1:end-boundary_pixel));
            WriteTifStack(Decon, [file_path '\Decon\', 'TileX_' num2str(x) '_TileY_' num2str(y) '_TileZ_' num2str(z) '.tif'], '16');
        end     
    end

    disp(['Tile_', num2str(tile), ' Done !']);  

    cTime5 = toc;
  
    end
end
fclose('all'); 
unloadlibrary(lib_name);
end

function Registration(lib_path, lib_name, file_path, filenameA, filenameB, GPU_C)

% % % load DLL lib
libFile = [lib_path, lib_name, '.dll'];
libHFile = [lib_path, lib_name, '.h'];

loadlibrary(libFile, libHFile);
% % % check lib functions
mFunctions = libfunctions(lib_name,'-full');

start_t = cputime; 
stackA = uint16(TIFFStack([file_path, filenameA, '.tif']));
stackB = uint16(TIFFStack([file_path, filenameB, '.tif']));
sizeA = size(stackA)
sizeB = size(stackB)

Input_matrix = [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1];  

regTrigger = 1; % whether or not do registration, 1: yes; 0: no
regMethod = 6 % registration method
                % 1: translation only; 
                % 2: rigid body; 
                % 3: 7 degrees of freedom (translation, rotation, scaling equally in 3 dimemsions)
                % 4: 9 degrees of freedom(translation, rotation, scaling); 
                % 5: 12 degrees of freedom; 
                % 6: rigid body first, then do 12 degrees of freedom
inputTmx = 1; % whether or not use input matrix, 1: yes; 0: no
tmxPtr = libpointer('singlePtr',Input_matrix');  % input matrix pointer
FTOL = 0.0001; % reg threshold
itLimit = 2500; % maximun iteration number for registration
subBgtrigger = 1; % subtract background (mean value of the image) for registration, 1: yes, 0: no

deviceNum = GPU_C; % GPU device
regRecords = zeros(1,11); % records and feedback of registration
regRecordsPtr = libpointer('singlePtr',regRecords); 

% % % % % **********create arguments **********
h_regB = libpointer('singlePtr',stackA); % registration feedback pointer: registered B stack

% % % input images and PSFs
tic;
h_tiffA = libpointer('singlePtr',stackA);
h_tiffB = libpointer('singlePtr',stackB);
tifSizeA = libpointer('uint32Ptr',sizeA); % image size pointer
tifSizeB = libpointer('uint32Ptr',sizeB); % image size pointer

cudaStatus = calllib(lib_name,'reg_3dgpu',h_regB, tmxPtr, h_tiffA, h_tiffB,tifSizeA,...
        tifSizeB, regMethod,inputTmx, FTOL, itLimit, subBgtrigger, deviceNum, regRecordsPtr);
    
% % % save images
Reg_B = uint16(reshape(h_regB.Value,sizeA));
WriteTifStack(Reg_B, [file_path filenameB 'Reg.tif'], '16');
Matrix = tmxPtr.Value;
dlmwrite([file_path, filenameB 'RegMat.txt'],Matrix', 'precision',6,'delimiter',' ');
% movefile('log.txt', [file_path, 'log_down', '.txt']);
unloadlibrary(lib_name);
end

function StitchingZ(file_path_in, file_path_out,CropStepXY, CropStepZ, tilex, tiley, tilez)

DownSamplingFactor = 4;
downsampling_mtrix = [1/DownSamplingFactor 0 0; 0 1/DownSamplingFactor 0; 0 0 1];
tform_4 = affine2d(downsampling_mtrix);

xSizeF = CropStepXY; ySizeF = CropStepXY; zSizeF = CropStepZ; %final image size for stiching
xSize = xSizeF + 128; ySize = ySizeF + 128; zSize = zSizeF + 128; %initial crop size 

A_size = [ySize, xSize, zSize];
 
cut1 = 20; %pixel    
cut = 25;
overlap = 128 - cut1 * 2 - cut * 2;
boundary_pixel = cut;

M_size = [ySize-2*cut-2*cut1, xSize-2*cut-2*cut1 (zSize-2*cut-2*cut1)*tilez-(tilez-1)*overlap];

ramp = linspace(0, 1, overlap);
ramp_start = repmat(ramp, [ySize-2*cut-2*cut1, 1, ySize-2*cut-2*cut1]);   
ramp_start = permute(ramp_start,[1,3,2]);

data_block = cell(6);
tic;

for x=1:tilex
    
 for y=1:tiley
    
    tic;
for z = 1:tilez %1:tilez
    filename = [file_path_in '\' 'TileX_' num2str(x-1) '_TileY_' num2str(y-1) '_TileZ_' num2str(z-1) '.tif'];
    data_block{z} = TIFFStack(filename);        
end
 
MergeZ = zeros(M_size);

overlap_bottom = [];

for z = 1:tilez %1:tilez
     stack = single(data_block{z});
     stack_crop = stack(boundary_pixel+1:end-boundary_pixel,boundary_pixel+1:end-boundary_pixel,boundary_pixel+1:end-boundary_pixel);
     z_start = (z-1)*(zSize-2*cut-2*cut1-overlap) + 1;
     z_end = z_start + (zSize-2*cut-2*cut1) - 1;
     MergeZ(:,:,z_start:z_end)= stack_crop;  
     overlap_top = stack_crop(:,:,1:overlap);
     OverlapZ = BlendZ(overlap_bottom, overlap_top,ramp_start,overlap);
     MergeZ(:,:, z_start:z_start+overlap-1) = OverlapZ;
     overlap_bottom = stack_crop(:,:,end-overlap+1:end);
end

Time1 =  toc
  
WriteTifStack(MergeZ, [file_path_out '\TileX_' num2str(x-1) '_TileY_' num2str(y-1) '_TileZ_0.tif'], '16');
WriteTifStack(MergeZ(1:5:end,1:5:end,1:5:end), [file_path_out '\Downsampling\TileX_' num2str(x-1) '_TileY_' num2str(y-1) '_TileZ_0.tif'], '16');

 Time2 =  toc
end
end
end


function StitchingXY(file_path_in, file_path_out, CropStepXY, CropStepZ, tilex, tiley, nz)
tile = 0;

DownSamplingFactor = 5;
downsampling_mtrix = [1/DownSamplingFactor 0 0; 0 1/DownSamplingFactor 0; 0 0 1];
tform_4 = affine2d(downsampling_mtrix);
% 

xSizeF = CropStepXY; ySizeF = CropStepXY; zSizeF = CropStepZ; %final image size for stiching
xSize = xSizeF + 128; ySize = ySizeF + 128; zSize = zSizeF + 128; %initial crop size 

A_size = [ySize, xSize, zSize];
 
cut1 = 20; %pixel    boundary value...
cut = 25;  % extra cut .......
overlap = 128 - cut1 * 2 - cut * 2;

tile = 0;

data_block = cell(tilex, tiley);
tic;
y_start = 1; %0
y_end = tiley;

x_start = 1; %0
x_end = tilex;

for y = y_start:y_end %1:tiley
     parfor x = x_start:x_end %tilex
          filename = [file_path_in '\TileX_' num2str(x-1) '_TileY_' num2str(y-1) '_TileZ_0.tif'];
           data_block{x, y} = TIFFStack(filename);
     end       
end
Time1 =  toc
    
tile = 0;
 parfor z =  1 : nz
     tic;
      MergeY = []; 
      for y= y_start:y_end%1:tiley
         MergeX = [];
        for x = x_start:x_end%1:tilex
            slice = single(data_block{x, y}(cut+1:end-cut,cut+1:end-cut,z));
           % slice = single(data_block{x, y}(:,:, z));
            MergeX = Blend2D(MergeX, slice,2, overlap, 3);
        end
           MergeY = Blend2D(MergeY, MergeX, 1, overlap, 3);
        
      end
    
     Final = uint16(MergeY);
     singlefile_out = strcat(file_path_out, '\MergedDecon_z_', num2str(z),'.tif');
     imwrite(Final, singlefile_out);
     Time2 = toc
     
%    if mod(z,DownSamplingFactor) == 1        
%         downsampling_data = imwarp(uint16(MergeY),tform_4);  
%         singlefile_out = strcat(output_file_path,'\MergeSmallRegion\DownSampling\MergedDecon_z', num2str(z),'.tif');
%         imwrite(downsampling_data, singlefile_out);
%    end

 end
end


