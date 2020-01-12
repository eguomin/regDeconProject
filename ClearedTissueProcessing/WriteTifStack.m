function WriteTifStack(Stack, Filename, BitsPerSample)

  w = warning('off', 'MATLAB:imagesci:tiffmexutils:libtiffWarning');
  warning('off', 'MATLAB:imagesci:tifftagsread:expectedTagDataFormat');
   
   t = Tiff(Filename, 'w');
   tagstruct.Software = 'MATLAB';
   tagstruct.ImageLength = size(Stack, 1);
   tagstruct.ImageWidth = size(Stack, 2);
   tagstruct.Compression = Tiff.Compression.None;
   %tagstruct.Compression = Tiff.Compression.LZW;        % compressed
   
   switch BitsPerSample
       case 'RGB'
            tagstruct.Photometric = Tiff.Photometric.RGB;   
            tagstruct.BitsPerSample = 8;
            tagstruct.SamplesPerPixel = 3;
            tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            for k = 1:size(Stack,3)
                t.setTag(tagstruct);
                t.write(squeeze(Stack(:, :, k, 1:3)));
                t.writeDirectory();
            end
             t.close();
      
       case '32'
            tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
            tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            tagstruct.SamplesPerPixel = 1;
            tagstruct.BitsPerSample =  32; 
            tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
            Stack = single(Stack);
            for k = 1:size(Stack, 3)
                t.setTag(tagstruct);
                t.write(Stack(:, :, k));
                t.writeDirectory();
            end
             t.close();
             
       case '16'
            tagstruct.Photometric = Tiff.Photometric.MinIsBlack; 
            tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            tagstruct.SamplesPerPixel = 1;
            tagstruct.BitsPerSample =  16; 
            tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
            Stack = uint16(Stack);
            for k = 1:size(Stack, 3)
               t.setTag(tagstruct);
               t.write(Stack(:, :, k));
               t.writeDirectory();
            end
             t.close();

       end
      warning(w);
end     
     