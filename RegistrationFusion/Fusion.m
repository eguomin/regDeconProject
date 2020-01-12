% Fusion.m: do registration and re for time-sequence images of two views,
% the two view images (input) are assumed to have isotripic pixel size and oriented 
% in the same direction.

% load raw data
[filename_data, path_data] = uigetfile('*.tif','Choose any one of raw data');
% load PSF data 
[filename_psf, path_psf] = uigetfile('*.tif','Choose any one of PSF image',path_data);

% set parameters
inputDialog = uifigure('Name','Set Parameters','Position',[300 500 400 320]);
% set registration mode
uilabel(inputDialog,'Text','Registration mode:', 'Position',[30 290 300 20]);
% Create a button group and radio buttons:
bg = uibuttongroup('Parent',inputDialog,...
    'Position',[50 170 300 120]);
rb1 = uiradiobutton(bg,'Text','translation only','Position',[10 90 200 20]);
rb2 = uiradiobutton(bg,'Text','rigid body','Position',[10 70 200 20]);
rb3 = uiradiobutton(bg,'Text','7 DOF','Position',[10 50 200 20]);
rb4 = uiradiobutton(bg,'Text','9 DOF','Position',[10 30 200 20]);
rb5 = uiradiobutton(bg,'Text','12 DOF','Position',[10 10 200 20]);
% for time points
uilabel(inputDialog,'Text','Enter iteration number: :', 'Position',[30 140 300 20]);
txa1 = uitextarea(inputDialog, 'Value','10', 'Position',[50 120 100 20]);
uilabel(inputDialog,'Text','Enter time points to be processed:', 'Position',[30 100 300 20]);
txa2 = uitextarea(inputDialog, 'Value','0-1', 'Position',[50 70 100 30]);
% buttons to confirm/cancel processing
pb1 = uibutton(inputDialog,'push', 'Position',[130 30 50 20],'Text','Yes',...
    'ButtonPushedFcn', @(pb1,event) pbYes(inputDialog,bg,txa1,txa2, path_data, path_psf));
pb2 = uibutton(inputDialog,'push', 'Position',[210 30 50 20],'Text','Cancel',...
    'ButtonPushedFcn', @(pb1,event) pbCancel(inputDialog));
function pbCancel(inputDialog)
    delete(inputDialog);
    disp('Processing cancelled!!!')
end
function pbYes(inputDialog,bg,txa1, txa2, path_data, path_psf)
    bgChoice = get(get(bg,'SelectedObject'), 'Text');
    itStr = get(txa1,'Value');
    itNum = str2num(itStr{1});
    timeStr = get(txa2,'Value');
    timepoints = strsplit(timeStr{1},'-');
    t1 = str2num(timepoints{1});
    t2 = str2num(timepoints{2});
    delete(inputDialog);
    
    
    disp('Initialize processing...');
    % % % load DLL lib
    libPath = '..\cudaLib\';
    libName = 'libapi';
    libFile = [libPath, libName, '.dll'];
    libHFile = [libPath, libName, '.h'];
    loadlibrary(libFile, libHFile);
    % % output folder
    path_output = [path_data, 'results\'];
    mkdir(path_output);
    
    % % images and PSFs
    stackA = single(ReadTifStack([path_data 'StackA_' num2str(t1) '.tif']));
    stackB = single(ReadTifStack([path_data 'StackB_' num2str(t1) '.tif']));
    sizeA = size(stackA);
    sizeB = size(stackB);
    h_tiffA = libpointer('singlePtr',stackA);
    h_tiffB = libpointer('singlePtr',stackB);
    tifSizeA = libpointer('uint32Ptr',sizeA); % image size pointer
    tifSizeB = libpointer('uint32Ptr',sizeB); % image size pointer
    
    PSFA = single(ReadTifStack([path_psf 'PSFA.tif']));
    PSFB = single(ReadTifStack([path_psf 'PSFB.tif']));
    PSFA_bp = single(ReadTifStack([path_psf 'PSFA_BP.tif']));
    PSFB_bp = single(ReadTifStack([path_psf 'PSFB_BP.tif']));
    sizePSF = size(PSFA);
    h_PSFA = libpointer('singlePtr',PSFA);
    h_PSFB = libpointer('singlePtr',PSFB);
    h_PSFA_bp = libpointer('singlePtr',PSFA_bp);
    h_PSFB_bp = libpointer('singlePtr',PSFB_bp);
    tifSizePSF = libpointer('uint32Ptr',sizePSF); % image size pointer
    % % % % % **********create arguments **********
    % % % results 
    h_decon = libpointer('singlePtr',stackA); % decon feedback pointer: decon result
    h_regB = libpointer('singlePtr',stackA); % registration feedback pointer: registered B stack
    % % % parameters
    gpuMemMode = 1; % 1: efficient GPU mode; 2: GPU memory-saved mode
    % % %  configurations
    regChoice = 4; %  2D MIP registration --> affine registration (input matrix disabled);
    switch(bgChoice)
        case 'translation only'
            affMethod = 1;
        case 'rigid body'
            affMethod = 2;
        case '7 DOF'
            affMethod = 3;
        case '9 DOF'
            affMethod = 4;
        case '12 DOF'
            affMethod = 7; % 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
    end
    flagTmx = 0; % whether or not use input matrix, 1: yes; 0: no
                % *** flagTmx: only if regChoice == 0, 2
    Tmx = [1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1];
    tmxPtr = libpointer('singlePtr',Tmx);  % input matrix pointer
    FTOL = 0.0001; % reg threshold
    itLimit = 2000; % maximun iteration number for registration
    deviceNum = 0; % GPU device: numbering from 0 by CUDA;
    verbose = 0; % show details during processing: does not work for MATLAB
    records = zeros(1,11);
    h_records = libpointer('singlePtr',records); % reg records and feedback
    tic;
    disp('Start processing...');
    for imgNum = t1:t2
        cTime1 = toc;
        disp(['...Processing image #: ' num2str(imgNum)]);
        stackA = single(ReadTifStack([path_data 'StackA_' num2str(imgNum) '.tif']));
        stackB = single(ReadTifStack([path_data 'StackB_' num2str(imgNum) '.tif']));
        h_tiffA.Value = stackA;
        h_tiffB.Value = stackB;
        if(imgNum~=t1) % use last registration matrix as input 
            flagTmx = 1;
            regChoice = 2;
            if(affMethod == 7)
                affMethod =5; % change to directly 12 DOF
            end
        end
        disp('... ... Performing registration ...');
        runStatus = calllib(libName,'reg3d',h_regB, tmxPtr, h_tiffA, h_tiffB,...
            tifSizeA, tifSizeB, regChoice, affMethod, ...
            flagTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, h_records);
        stackB_reg = reshape(h_regB.Value,sizeA);
        WriteTifStack(stackB_reg, [path_output, 'StackB_reg_', num2str(imgNum), '.tif'], 16);
        
        disp('... ... Performing deconvolution ...');
        flagConstInitial = 0;
        flagUnmatch = 0;
        runStatus = calllib(libName,'decon_dualview', h_decon, h_tiffA, h_regB, tifSizeA, h_PSFA, h_PSFB,...
            tifSizePSF, flagConstInitial, itNum, deviceNum, gpuMemMode, verbose, h_records, flagUnmatch, h_PSFA_bp, h_PSFB_bp);
        stack_decon = reshape(h_decon.Value,sizeA);
        WriteTifStack(stack_decon, [path_output, 'Decon_', num2str(imgNum), '.tif'], 16);
        cTime2 = toc;
        disp(['... ... Time cost for current image: ' num2str(cTime2 - cTime1) ' s']);
    end
    % % % unload DLL lib 
    unloadlibrary(libName);
    cTime3 = toc;
    disp(['... Total time cost: ' num2str(cTime3) ' s']);
    disp('Processing completed !!!');
end
