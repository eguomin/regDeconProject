clear all;

w = warning('off', 'MATLAB:imagesci:tiffmexutils:libtiffWarning');
warning('off', 'MATLAB:imagesci:tifftagsread:expectedTC1_TranslatedagDataFormat');

file_path = uigetdir();
%file_path = 'E:\TedBrain\';

shift_pixel = 2;  % set step size of stage scanning 2 um here
shear = shift_pixel/(6.5/17);  % here 6.5 is the pixel size of the camera, 17 is the Mag of the detection optics

% file_nameA = strcat(file_path, 'StackA_StageMode.tif');
% file_nameB = strcat(file_path, 'StackB_StageMode.tif');

file_nameA = strcat(file_path, '\','Stage_SPIMA_C1.tif');
file_nameB = strcat(file_path, '\','Stage_SPIMB_C1.tif');

StackA_file_path = [file_path, 'SheetModeA'];
if ~exist(StackA_file_path, 'dir')
  mkdir(StackA_file_path);
end

StackA_Down_file_path = [StackA_file_path, 'Downsampling'];
if ~exist(StackA_Down_file_path, 'dir')
  mkdir(StackA_Down_file_path);
end

StackB_file_path = [file_path, 'SheetModeB'];
if ~exist(StackB_file_path, 'dir')
  mkdir(StackB_file_path);
end

StackB_Down_file_path = [StackB_file_path, 'Downsampling'];
if ~exist(StackB_Down_file_path, 'dir')
  mkdir(StackB_Down_file_path);
end


start_t = cputime;



theta = 45;
DownSamplingFactor = 5;

stagescanning_matix = [1 0 0; shear 1 0; 0 0 1];
zinterpolation_mtrix = [1 0 0; 0 shear 0; 0 0 1];
downsampling_mtrix = [1/DownSamplingFactor 0 0; 0 1/DownSamplingFactor 0; 0 0 1];
rotation_matrix = [cosd(theta) sind(theta) 0; -sind(theta) cosd(theta) 0; 0 0 1];

options.color     = false;
options.compress  = 'no';
options.message   = false;
options.append    = false;
options.overwrite = true;
options.big       = true;

end_t = cputime - start_t;

tform_1 = affine2d(stagescanning_matix);
tform_2 = affine2d(zinterpolation_mtrix);
tform_3 = affine2d(rotation_matrix);
tform_4 = affine2d(downsampling_mtrix);

dataA = TIFFStack(file_nameA);
[ny, nx, nz] = size(dataA);

nz1 = shear * nz;
nx1 = shear * nz + nx;

% nz2_bottom = ceil(nz1 * 1.4142 /2);
% nz2_top = round(nz2_bottom - nx/1.4142);
% nz2_top = nz2_bottom - round((nz2_bottom - nz2_top)/20)*20 +1;
% 
% nx2_start = nz2_bottom - nz2_top;
% nx2_end =  ceil(nz1 * 1.4142) - nx2_start;
% nx2_end = round((nx2_end - nx2_start)/20)*20 + nx2_start - 1;


parfor number = 1:ny  %parfor
   k = number;
   if mod(k,50)==0
       disp('.')
   end
   xz_data = squeeze(dataA(k,:,:))'-100;
   xz_data = imwarp(xz_data,tform_1);
%    xz_data = imwarp(xz_data(:,nx+1:end),tform_2);
   xz_data = imwarp(xz_data,tform_2);
   xz_data = imwarp(xz_data,tform_3);
%    merge_data = xz_data;
   merge_data = xz_data(2870:7750,3980:7655);
%    merge_data = xz_data(3980:7655,2870:7750);
%    merge_data = xz_data(nz2_top:nz2_bottom,nx2_start:nx2_end);
    % if A view, need to flip; if B view, not need to flip.
%    merge_data = flip(merge_data,1);
   merge_data=rot90(merge_data,1);
   singlefile_out = strcat(StackA_file_path, '\', 'slice', num2str(k),'.tif');
   imwrite(merge_data, singlefile_out);

   if mod(k,DownSamplingFactor) == 1
        downsampling_data = imwarp(merge_data,tform_4);
        singlefile_out = strcat(StackA_Down_file_path, '\', 'slice_down_', num2str(k),'.tif');
        imwrite(downsampling_data, singlefile_out);
   end

end



dataB = TIFFStack(file_nameB);
[ny, nx, nz] = size(dataB);

nz1 = shear * nz;
nx1 = shear * nz + nx;

% nz2_bottom = ceil(nz1 * 1.4142 /2) ;
% nz2_top = round(nz2_bottom - nx/1.4142);
% nz2_top = nz2_bottom - round((nz2_bottom - nz2_top)/20)*20 +1 ;
% 
% nx2_start = nz2_bottom - nz2_top;
% nx2_end =  ceil(nz1 * 1.4142) - nx2_start;
% nx2_end = round((nx2_end - nx2_start)/20)*20 + nx2_start - 1;


parfor number = 1000 : 1500 %ny
   k = number;
   if mod(k,50)==0
       disp('.')
   end
   xz_data = squeeze(dataB(k,:,:))'-100; 
   xz_data = imwarp(xz_data,tform_1);
   xz_data = imwarp(xz_data,tform_2);
%    xz_data = imwarp(xz_data(:,nx+1:end),tform_2);
   xz_data = imwarp(xz_data,tform_3);   
%    merge_data = xz_data;
   merge_data = xz_data(2870:7750,3980:7655);
%    merge_data = xz_data(nz2_top:nz2_bottom,nx2_start:nx2_end);
  % if A view, need to flip; if B view, not need to flip. 
   merge_data = flip(merge_data,1);
   merge_data=rot90(merge_data,-1);
   singlefile_out = strcat(StackB_file_path, '\', 'slice', num2str(k),'.tif');
   imwrite(merge_data, singlefile_out);
    
   if mod(k,DownSamplingFactor) == 1        
        downsampling_data = imwarp(merge_data,tform_4);  
        singlefile_out = strcat(StackB_Down_file_path, '\', 'slice_down_', num2str(k),'.tif');
        imwrite(downsampling_data, singlefile_out);
   end
    
end
 end_t = cputime - start_t
 
