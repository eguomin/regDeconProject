% Registration.m: do registration for time-sequence images of two views,
% the two view images are assumed to have same pixel size and oriented 
% in the same direction.

% load raw data
[filename_data, path_data] = uigetfile('*.tif','Choose any one of raw data');

% set parameters
inputDialog = uifigure('Name','Set Parameters','Position',[300 500 400 280]);
% set registration mode
uilabel(inputDialog,'Text','Registration mode:', 'Position',[30 250 300 20]);
% Create a button group and radio buttons:
bg = uibuttongroup('Parent',inputDialog,...
    'Position',[50 130 300 120]);
rb1 = uiradiobutton(bg,'Text','translation only','Position',[10 90 200 20]);
rb2 = uiradiobutton(bg,'Text','rigid body','Position',[10 70 200 20]);
rb3 = uiradiobutton(bg,'Text','7 DOF','Position',[10 50 200 20]);
rb4 = uiradiobutton(bg,'Text','9 DOF','Position',[10 30 200 20]);
rb5 = uiradiobutton(bg,'Text','12 DOF','Position',[10 10 200 20]);
% for time points
uilabel(inputDialog,'Text','Enter time points to be processed:', 'Position',[30 100 300 20]);
txa = uitextarea(inputDialog, 'Value','0-1', 'Position',[50 70 100 30]);
% buttons to confirm/cancel processing
pb1 = uibutton(inputDialog,'push', 'Position',[130 30 50 20],'Text','Yes',...
    'ButtonPushedFcn', @(pb1,event) pbYes(inputDialog,bg,txa, path_data));
pb2 = uibutton(inputDialog,'push', 'Position',[210 30 50 20],'Text','Cancel',...
    'ButtonPushedFcn', @(pb1,event) pbCancel(inputDialog));
function pbCancel(inputDialog)
    delete(inputDialog);
    disp('Processing cancelled!!!')
end
function pbYes(inputDialog,bg,txa, path_data)
    bgChoice = get(get(bg,'SelectedObject'), 'Text');
    timeStr = get(txa,'Value');
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
    stackA = single(ReadTifStack([path_data 'StackA_' num2str(t1) '.tif']));
    stackB = single(ReadTifStack([path_data 'StackB_' num2str(t1) '.tif']));
    sizeA = size(stackA);
    sizeB = size(stackB);
    h_tiffA = libpointer('singlePtr',stackA);
    h_tiffB = libpointer('singlePtr',stackB);
    % % % % % **********create arguments **********
    % % % results 
    h_regB = libpointer('singlePtr',stackA); % registration feedback pointer: registered B stack
    % % % parameters
    tifSizeA = libpointer('uint32Ptr',sizeA); % image size pointer
    tifSizeB = libpointer('uint32Ptr',sizeB); % image size pointer
    gpuMemMode = 1; % 1: efficient GPU mode; 2: GPU memory-saved mode
    % % %  configurations
    regChoice = 4; % *** registration choice: regChoice
                % 0: no phasor or affine registration; if flagTmx is true, transform d_img2 based on input matrix;
                % 1: phasor registraion (pixel-level translation only);
                % 2: affine registration (with or without input matrix);
                % 3: phasor registration --> affine registration (input matrix disabled);
                % 4: 2D MIP registration --> affine registration (input matrix disabled);
    
    switch(bgChoice)
        % affine registration method: only if regChoice == 2, 3, 4
                % 0: no registration; 
                % 1: translation only; 
                % 2: rigid body; 
                % 3: 7 degrees of freedom (translation, rotation, scaling equally in 3 dimemsions)
                % 4: 9 degrees of freedom(translation, rotation, scaling); 
                % 5: 12 degrees of freedom; 
                % 6: rigid body first, then 12 degrees of freedom
                % 7: translation, rigid body, 9 degrees of freedom and then 12 degrees of freedom
        case 'translation only'
            affMethod = 1;
        case 'rigid body'
            affMethod = 2;
        case '7 DOF'
            affMethod = 3;
        case '9 DOF'
            affMethod = 4;
        case '12 DOF'
            affMethod = 7;
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
    regRecords = libpointer('singlePtr',records); % reg records and feedback
    tic;
    disp('Start registration...');
    for imgNum = t1:t2
        cTime1 = toc;
        disp(['...Processing image #: ' num2str(imgNum)]);
        stackA = single(ReadTifStack([path_data 'StackA_' num2str(imgNum) '.tif']));
        stackB = single(ReadTifStack([path_data 'StackB_' num2str(imgNum) '.tif']));
        h_tiffA.Value = stackA;
        h_tiffB.Value = stackB;
        % % % % % run registration function: reg3d
        runStatus = calllib(libName,'reg3d',h_regB, tmxPtr, h_tiffA, h_tiffB,...
            tifSizeA, tifSizeB, regChoice, affMethod, ...
            flagTmx, FTOL, itLimit, deviceNum, gpuMemMode, verbose, regRecords);
        stackB_reg = reshape(h_regB.Value,sizeA);
        WriteTifStack(stackB_reg, [path_output, 'StackB_reg_', num2str(imgNum), '.tif'], 16);
        cTime2 = toc;
        disp(['... ... Time cost for current image: ' num2str(cTime2 - cTime1) ' s']);
    end
    cTime3 = toc;
    % % % unload DLL lib 
    unloadlibrary(libName);
    disp(['... Total time cost: ' num2str(cTime3) ' s']);
    disp('Registration completed !!!');
end
