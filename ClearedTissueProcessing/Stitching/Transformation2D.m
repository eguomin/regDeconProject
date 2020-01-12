function RegB = Transformation2D(Tmx, Stripe_Obj, nx, ny, nz, z1, z2)

lib_path = 'C:\Yicong_Program\CUDA_DLL\RegTest\DLLs\';
lib_name = 'libapi';
libFile = [lib_path, lib_name, '.dll'];
libHFile = [lib_path, lib_name, '.h'];

loadlibrary(libFile, libHFile);
mFunctions = libfunctions(lib_name,'-full');

minz = z1 + Tmx(4,3);
maxz = z2 + Tmx(4,3);

% maxz = max(min(max(B(:))+1,nz),1);
% minz = min(max(min(B(:))+1,1),nz);

Tmx(4,3) = minz - floor(minz);
StackB = Stripe_Obj.getRegion([1 1 floor(minz)], [ny nx ceil(maxz)]);

% RegB = imtranslate(StackB, [Tmx(4,1), Tmx(4,2), Tmx(4,3)]);
% RegB = RegB(:,:,1:z2-z1+1);

sizeB = size(StackB);
Transformed = StackB(:,:,1:z2-z1+1);
sizeO = [sizeB(1), sizeB(2),z2-z1+1];  
deviceNum = 0; % GPU device
tmxPtr = libpointer('singlePtr',Tmx);  % input matrix pointer
h_output = libpointer('singlePtr',Transformed); % registration feedback pointer: registered B stack
h_input = libpointer('singlePtr',StackB);
RegSize = libpointer('uint32Ptr',sizeO); % A after transformation
InputSize = libpointer('uint32Ptr',sizeB); % B input
cudaStatus = calllib(lib_name,'affinetrans_3dgpu', h_output, tmxPtr, h_input, RegSize, InputSize, deviceNum);
RegB = uint16(reshape(h_output.Value,sizeO));


end%