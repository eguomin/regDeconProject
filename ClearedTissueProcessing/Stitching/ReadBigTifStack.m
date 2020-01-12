function Stack = ReadBigTifStack(FileTif, slices, zoom)

  w = warning('off', 'MATLAB:imagesci:tiffmexutils:libtiffWarning');
  warning('off', 'MATLAB:imagesci:tifftagsread:expectedTagDataFormat');
  InfoImage = imfinfo(FileTif);
  
  if length(InfoImage)>1
      disp('it is a regular TIFF stack, less than 4GB');
      Depth = length(InfoImage);
      Width = InfoImage(1).Width;
      Height = InfoImage(1).Height;
      
      if nargin == 1
          slices = 1:Depth;
          zoom = 1;
      elseif nargin == 2
          zoom = 1;
      end     
      Stack = zeros(ceil(zoom*Height),ceil(zoom*Width),length(slices));
      % load with traditinal method:
      TifLink = Tiff(FileTif, 'r');
      if zoom == 1
        for i = 1:length(slices)
            TifLink.setDirectory(slices(i));
            Stack(:,:,i) = TifLink.read();
        end
      else
          for i = 1:length(slices)
            TifLink.setDirectory(slices(i));
            Stack(:,:,i) = imresize(TifLink.read(),zoom);
          end
      end
        TifLink.close();
  
  elseif length(InfoImage)==1
      disp('it is a >4GB Big TIFF Stack');
      Width = InfoImage.Width;
      Height = InfoImage.Height; 
      Depth = round(InfoImage.FileSize/(Width*Height*InfoImage.BitDepth/16*2));
      if nargin == 1
          slices = 1:Depth;
          zoom = 1;
      elseif nargin == 2
          zoom = 1;
      end
    % Use low-level File I/O to read the file
    fp = fopen(FileTif, 'rb');
    fseek(fp, InfoImage.StripOffsets, 'bof');
    Stack = zeros(ceil(zoom*Height),ceil(zoom*Width),length(slices));
 
    if zoom == 1
        for i=1:length(slices)
            fseek(fp, InfoImage.StripOffsets + (slices(i)-1)*Width*Height*2, 'bof');
            Stack(:,:,i) = fread(fp,[Width Height], 'uint16', 0, 'ieee-be')';
        end
        
    else
        for i = 1:length(slices)
            fseek(fp, InfoImage.StripOffsets + (slices(i)-1)*Width*Height*2, 'bof');
            Stack(:,:,i) = imresize(fread(fp,[Width Height], 'uint16', 0, 'ieee-be')',zoom);
        end
    end
    fclose(fp);
  end
    warning(w);
end

%Another faster tiff stack loader
% InfoImage = imfinfo(FileTif);
% mImage = InfoImage(1).Width;
% nImage = InfoImage(1).Height;
% NumberImages = length(InfoImage);
% FinalImage=zeros(nImage,mImage,NumberImages,'uint16');
% FileID = tifflib('open',FileTif,'r');
% rps = tifflib('getField',FileID,Tiff.TagID.RowsPerStrip);
%  
% for i=1:NumberImages
%   tifflib('setDirectory',FileID,i);
%   % Go through each strip of data.
%   rps = min(rps,nImage);
%   for r = 1:rps:nImage
%      row_inds = r:min(nImage,r+rps-1);
%      stripNum = tifflib('computeStrip',FileID,r);
%      FinalImage(row_inds,:,i) = tifflib('readEncodedStrip',FileID,stripNum);
%   end
% end
% tifflib('close',FileID);