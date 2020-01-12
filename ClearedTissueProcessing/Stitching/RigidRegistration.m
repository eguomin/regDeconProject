function [M, r, RegB] = RigidRegistration(stackB, stackA, dll_path)

[lib_path,lib_name] = fileparts(dll_path);
% lib_path = 'C:\Yicong_Program\CUDA_DLL\spimfusion_DLL\';
% lib_name = 'libapi_0605';
libFile = [lib_path, '\', lib_name, '.dll'];
libHFile = [lib_path, '\', lib_name, '.h'];
loadlibrary(libFile, libHFile);
mFunctions = libfunctions(lib_name,'-full');

Initial_matrix = [1, 0, 0,0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1]';
sizeA = size(stackA);
sizeB = size(stackB);
regMethod = 1; % registration method
inputTmx = 0; % whether or not use input matrix, 1: yes; 0: no
tmxPtr = libpointer('singlePtr',Initial_matrix);  % input matrix pointer
FTOL = 0.0001; % reg threshold
itLimit = 3000; % maximun iteration number for registration
subBgtrigger = 0; % subtract background (mean value of the image) for registration, 1: yes, 0: no

deviceNum = 0; % GPU device
regRecords = zeros(1,11);
regRecordsPtr = libpointer('singlePtr',regRecords); % Time cost pointer

% % % % % **********create arguments **********
h_regB = libpointer('singlePtr',stackA); % registration feedback pointer: registered B stack
% % % input images and PSFs
h_tiffA = libpointer('singlePtr',stackA);
h_tiffB = libpointer('singlePtr',stackB);
tifSizeA = libpointer('uint32Ptr',sizeA); % image size pointer
tifSizeB = libpointer('uint32Ptr',sizeB); % image size pointer

cudaStatus = calllib(lib_name,'reg_3dgpu',h_regB, tmxPtr, h_tiffA, h_tiffB,...
        tifSizeA, tifSizeB, regMethod, inputTmx, FTOL, itLimit, subBgtrigger, deviceNum, regRecordsPtr); 
% % % save images
RegB = uint16(reshape(h_regB.Value,sizeA));
Matrix = tmxPtr.Value;
unloadlibrary(lib_name); 


M(1) = Matrix(4,1);
M(2) = Matrix(4,2);
M(3) = Matrix(4,3);

sy = sizeA(1);
sx = sizeA(2);
sz = sizeA(3);
shiftB=[];
shiftA=[];

if M(1)>0 & M(1)<=sy & M(2)>0 & M(2)<=sx & M(3)>0 & M(3)<=sz % > > > a a a
          shiftB = RegB(1:end-M(1), 1:end-M(2),1:end-M(3));
          shiftA = stackA(1:end-M(1), 1:end-M(2),1:end-M(3));      
    elseif M(1)<=0 & M(1)>-sy & M(2)<=0 & M(2)>-sx & M(3)<=0 & M(3)>-sz % < < < b b b         
          shiftB = RegB(-M(1)+1:end,-M(2)+1:end,-M(3)+1:end);
          shiftA = stackA(-M(1)+1:end,-M(2)+1:end,-M(3)+1:end);  
    elseif M(1)>0 & M(1)<=sy & M(2)<=0 & M(2)>-sx & M(3)<=0 & M(3)>-sz % > < < a b b
        shiftB = RegB(1:end-M(1),-M(2)+1:end, -M(3)+1:end);
        shiftA = stackA(1:end-M(1),-M(2)+1:end, -M(3)+1:end);
    elseif M(1)<=0 & M(1)>-sy & M(2)>0 & M(2)<=sx & M(3)>0 & M(3)<=sz % < > > b a a
        shiftB = RegB(-M(1)+1:end,1:end-M(2),1:end-M(3));
        shiftA = stackA(-M(1)+1:end,1:end-M(2),1:end-M(3));
     elseif M(1)>0 & M(1)<=sy & M(2)>0 & M(2)<=sx & M(3)<=0 & M(3)>-sz % > > < a a b
         shiftB = RegB(1:end-M(1),1:end-M(2),-M(3)+1:end);
         shiftA = stackA(1:end-M(1),1:end-M(2),-M(3)+1:end);
    elseif M(1)<=0 & M(1)>-sy & M(2)>0 & M(2)<=sx & M(3)<=0 & M(3)>-sz% < > < b a b
        shiftB = RegB(-M(1)+1:end,1:end-M(2),-M(3)+1:end);
        shiftA = stackA(-M(1)+1:end,1:end-M(2),-M(3)+1:end);
    elseif M(1)>0 & M(1)<=sy & M(2)<=0 & M(2)>-sx & M(3)>0 & M(3)<=sz  % > < > a b a
        shiftB = RegB(1:end-M(1),-M(2)+1:end,1:end-M(3));
        shiftA = stackA(1:end-M(1),-M(2)+1:end,1:end-M(3));
    elseif M(1)<=0 & M(1)>sy & M(2)<=0 & M(2)>-sx & M(3)>0 & M(3)<=sz % < < > b b a
        shiftB = RegB(-M(1)+1:end, -M(2)+1:end, 1:end-M(3));
        shiftA = stackA(-M(1)+1:end, -M(2)+1:end, 1:end-M(3));
end

  shiftA = single(shiftA);
  shiftB = single(shiftB);
  shiftA = shiftA - mean(shiftA(:));
  shiftB = shiftB - mean(shiftB(:));
  Coef = shiftA.*shiftB/(std(shiftA(:))*std(shiftB(:)));
  [ny, nx, nz] = size(shiftA);
  r = sum(Coef(:))/(ny*nx*nz);

  
end