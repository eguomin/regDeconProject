function WriteBigStack(data, outTiff, Bit, IFD)

tileSize = [640 640];
Height = size(data,1);
Width = size(data,2);

tagstruct.Software = 'MATLAB';
 
tagstruct.ImageLength = Height;
tagstruct.ImageWidth = Width;
tagstruct.TileLength = tileSize(1);
tagstruct.TileWidth = tileSize(2);
tagstruct.Compression = Tiff.Compression.None;
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.SamplesPerPixel = 1;

tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    
switch Bit
   case '32'           
            tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            tagstruct.BitsPerSample =  32; 
            tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
         %   tagstruct.RowsPerStrip = 256;
            data = single(data);          
                         
   case '16'
            tagstruct.BitsPerSample =  16; 
            tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
          %  tagstruct.RowsPerStrip = 256;
            data = uint16(data);
end
   
    outTiff.setTag(tagstruct);

    % Loop through the input image and write out tiles  
    for tiledRowInd = (1:ceil(Height/tileSize(1)))-1
        rows = [tiledRowInd*tileSize(1)+1, tiledRowInd*tileSize(1)+tileSize(1)];
        rows(2) = min(rows(2), Height);
        for tiledColInd = (1:ceil(Width/tileSize(2)))-1
            cols = [tiledColInd*tileSize(1)+1, tiledColInd*tileSize(1)+tileSize(1)];
            cols(2) = min(cols(2), Width);
            tileData = data(rows(1):rows(2), cols(1):cols(2));
            tileInd = outTiff.computeTile([rows(1), cols(1)]);
            outTiff.writeEncodedTile(tileInd, tileData);
        end
    end 
     
   %  Write entire image
   %  outTiff.write(data);
     outTiff.writeDirectory();
   % outTiff.close();
end
