function varargout = Stitch_GUI(varargin)
% STITCH_GUI MATLAB code for Stitch_GUI.fig
%      STITCH_GUI, by itself, creates a new STITCH_GUI or raises the existing
%      singleton*.
%
%      H = STITCH_GUI returns the handle to a new STITCH_GUI or the handle to
%      the existing singleton*.
%
%      STITCH_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in STITCH_GUI.M with the given input arguments.
%
%      STITCH_GUI('Property','Value',...) creates a new STITCH_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Stitch_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Stitch_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Stitch_GUI

% Last Modified by GUIDE v2.5 19-Dec-2019 21:20:07

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Stitch_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @Stitch_GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Stitch_GUI is made visible.
function Stitch_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Stitch_GUI (see VARARGIN)

% Choose default command line output for Stitch_GUI
handles.output = hObject;

TileInfo = [1, 1, 1, 1; 0.2, 0.2, 0.2, NaN; 1, 1, 1, NaN]';
set(handles.Table,'Data', TileInfo);
warning('off','all')

guidata(hObject, handles);

% Update handles structure


% UIWAIT makes Stitch_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Stitch_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object creation, after setting all properties.
function Table_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Table (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called




function TileName_Callback(hObject, eventdata, handles)
% hObject    handle to TileName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TileName as text
%        str2double(get(hObject,'String')) returns contents of TileName as a double


% --- Executes during object creation, after setting all properties.
function TileName_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TileName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in SelectFiles.
function SelectFiles_Callback(hObject, eventdata, handles)
% hObject    handle to SelectFiles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
guidata(hObject, handles);
tile_path  = handles.Path.String;
[handles.FileName,handles.path] = uigetfile([tile_path, '\*.tif'],'MultiSelect','on');
handles.FileNumber = length(handles.FileName);
set(handles.FileInfo, 'Min', 0, 'Max', 2);

for i=1:handles.FileNumber
    handles.TileStack{i} = TIFFStack([handles.path, handles.FileName{i}], false, [], true);
    ndim = size(handles.TileStack{i});    
    if length(ndim) == 3
        ny{i} = ndim(1);
        nx{i} = ndim(2);
        nz{i} = ndim(3);
        nc{i} = 1;
        Flag = 1;   
        totalsize{i} = ndim(1)*ndim(2)*ndim(3)*2;
        info{i} = [handles.FileName{i}, '   ', num2str(nx{i}), ' x ', num2str(ny{i}), ' x ', num2str(nz{i}), ' x ', num2str(nc{i})];
    elseif length(ndim) == 5
        ny{i} = ndim(1);
        nx{i} = ndim(2);
        nz{i} = ndim(4);
        nc{i} = ndim(5);
        Flag = 1;
        totalsize{i} = ndim(1)*ndim(2)*ndim(4)*ndim(5);
        info{i} = [handles.FileName{i}, '   ', num2str(nx{i}), ' x ', num2str(ny{i}), ' x ', num2str(nz{i}), ' x ', num2str(nc{i})];
  
    else
        Flag = 0;
    end
            
end

if Flag == 1
    TileSize = get(handles.Table, 'data');
    TileSize(2,1) = handles.FileNumber;
    TileSize(1,1) = 1;
    TileSize(3,1) = 1;
    TileSize(4,1) = nc{1};
    handles.scaleX = max(round(max(cell2mat(nx))/512),1);
    handles.scaleY = max(round(max(cell2mat(ny))/512),1);
    handles.scaleZ = max(round(max(cell2mat(nz))/512),1); 
    set(handles.FileInfo,'String', info);
    if length(ndim) ~= 3
        handles.FileInfo.String{end+2} = ['This stitcher can only process 3D TIFF Stack !!!'];
    end
    
    datasize = sum(cell2mat(totalsize))/(10^9);
    
    handles.FileInfo.String{end+2} = ['total ', num2str(handles.FileNumber), ' files, ' num2str(datasize), ' GB.  ', 'Modify the order now !'];
    set(handles.Table,'Data', TileSize);

elseif Flag == 0
    handles.FileInfo.String{end+2} = ['File format is not supported, please check'];
end

handles.Path.String = handles.path;

guidata(hObject, handles);

% Hints: contents = cellstr(get(hObject,'String')) returns SelectFiles contents as cell array
%        contents{get(hObject,'Value')} returns selected item from SelectFiles

function FileInfo_Callback(hObject, eventdata, handles)
% hObject    handle to FileInfo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of FileInfo as text
%        str2double(get(hObject,'String')) returns contents of FileInfo as a double


% --- Executes during object creation, after setting all properties.
function FileInfo_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FileInfo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in ViewTiles.
function ViewTiles_Callback(hObject, eventdata, handles)
% hObject    handle to ViewTiles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
tic
guidata(hObject, handles);
FileName = get(handles.FileInfo,'String');
Flag = get(handles.LoadDataMemory,'Value');
TileSize = get(handles.Table, 'data');
TileX = TileSize(1,1);
TileY = TileSize(2,1);
TileZ = TileSize(3,1);
TileC = TileSize(4,1);
OverlapX = TileSize(1,2);
OverlapY = TileSize(2,2);
OverlapZ = TileSize(3,2);

k = 0;
N = 1;  
for c=1:TileC
    for z=1:TileZ     
         k=k+2;
        for y = 1:TileY   
            k = k + 1;
            FileNameX = FileName{k};
            TileName = strsplit(FileNameX,char(9));
              for x = 1:TileX
                  handles.TileHandle{N} = TIFFStack([handles.path, TileName{x}], false, [], true);
                  N = N + 1;
               if Flag == 1
                   tic;
                      handles.TileData{N} = ReadBigTifStack([handles.path, TileName{x}]);
                    t = toc;
                       handles.FileInfo.String{end+2} = [TileName{x}, ' is loaded to memory,', ' it takes ', num2str(t), ' s' ];
                       drawnow();
               end       
           end
          
        end
    end
end
    
[ny, nx, nz] = size(handles.TileHandle{1});
sizeX = nx * TileX;
sizeY = ny * TileY;
sizeZ = nz * TileZ;
sn = nz;
set(handles.posSlider_RawData,'Min',1);
set(handles.posSlider_RawData,'Max',sn+1);
set(handles.FirstSlice, 'String', ['z: ', num2str(1)]);
set(handles.LastSlice, 'String', ['z: ', num2str(sn)]);
pos = round(sn/2);
dataStitchXY = [];
z = 1; 
c = 1;
for y = 1:TileY
   dataStitchX = [];
        for x = 1:TileX
            N = x + (y-1)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
            if Flag==1
                dataPart = handles.TileData{N}(:,:,pos);
            else
                dataPart = uint16(handles.TileHandle{N}(:,:,pos));
            end
            dataStitchX = cat(2, dataStitchX, dataPart);
        end
   dataStitchXY = cat(1, dataStitchXY, dataStitchX);
end

set(handles.CurrentSlice, 'String', num2str(pos));
set(handles.posSlider_RawData,'Value',pos);
set(handles.posSlider_RawData,'SliderStep',[1/(sn-1), 0.01]);

im_max = max(dataStitchXY(:));
im_min = min(dataStitchXY(:));
handles.hfig = imtool(dataStitchXY,[im_min im_max]);
handles.imthandle = findobj(handles.hfig,'type','image');
Current_title = sprintf('Before Merge: Slice-# %d', pos);
set(handles.hfig, 'Name', Current_title);

guidata(hObject, handles);
toc

% --- Executes on slider movement.
function posSlider_RawData_Callback(hObject, eventdata, handles)
% hObject    handle to posSlider_RawData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
Flag = get(handles.LoadDataMemory,'Value');
imin = get(handles.posSlider_RawData,'Min');
imax = get(handles.posSlider_RawData,'Max');
pos = round(get(handles.posSlider_RawData,'Value'));
TileSize = get(handles.Table, 'data');
TileX = TileSize(1,1);
TileY = TileSize(2,1);
TileZ = TileSize(3,1);
TileC = TileSize(4,1);

z = get(handles.TileNumber,'Value');
c = get(handles.ColorNumber,'Value');

if (pos == imax)
    set(handles.CurrentSlice,'String','MP-Z');
    set(handles.posSlider_RawData,'Value',pos);
    dataStitchXY = []; 
    for y = 1:TileY
            dataStitchX = [];
            for x = 1:TileX
                  N = x + (y-1)*TileX + (z-1)*TileX*TileY * (c-1)*TileX*TileY*TileZ;
                if Flag==1
                    dataPart = max(handles.TileData{N},[],3);
                else
                    dataPart = uint16(handles.TileHandle{N}(:,:,pos-1));
                end
                dataStitchX = cat(2, dataStitchX, dataPart);
            end
            dataStitchXY = cat(1, dataStitchXY, dataStitchX);
    end

    set(handles.imthandle, 'CData', dataStitchXY);
    Current_title = sprintf('Before Merge: Z Maximun Intensity');
    set(handles.hfig, 'Name', Current_title);
elseif (imin <= pos < imax)
    set(handles.CurrentSlice,'String',num2str(pos));
    set(handles.posSlider_RawData,'Value',pos);
    
    dataStitchXY = [];
    for y = 1:TileY
            dataStitchX = [];
            for x = 1:TileX
                 N = x + (y-1)*TileX + (z-1)*TileX*TileY * (c-1)*TileX*TileY*TileZ;
                if Flag==1
                    dataPart = handles.TileData{N}(:,:,pos);
                else
                    dataPart = uint16(handles.TileHandle{N}(:,:,pos-1));
                end
                dataStitchX = cat(2, dataStitchX, dataPart);
            end
            dataStitchXY = cat(1, dataStitchXY, dataStitchX);
    end
   
    set(handles.imthandle, 'CData', dataStitchXY);
    Current_title = sprintf('Before Merge: Slice-# %d', pos);
    set(handles.hfig, 'Name', Current_title);
end

guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function posSlider_RawData_CreateFcn(hObject, eventdata, handles)
% hObject    handle to posSlider_RawData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function FirstSlice_Callback(hObject, eventdata, handles)
% hObject    handle to FirstSlice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of FirstSlice as text
%        str2double(get(hObject,'String')) returns contents of FirstSlice as a double


% --- Executes during object creation, after setting all properties.
function FirstSlice_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FirstSlice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function LastSlice_Callback(hObject, eventdata, handles)
% hObject    handle to LastSlice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of LastSlice as text
%        str2double(get(hObject,'String')) returns contents of LastSlice as a double


% --- Executes during object creation, after setting all properties.
function LastSlice_CreateFcn(hObject, eventdata, handles)
% hObject    handle to LastSlice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function CurrentSlice_Callback(hObject, eventdata, handles)
guidata(hObject, handles);
pos = str2num(get(handles.CurrentSlice, 'String'));
set(handles.posSlider_RawData,'Value',pos);
Flag = get(handles.LoadDataMemory,'Value');
imin = get(handles.posSlider_RawData,'Min');
imax = get(handles.posSlider_RawData,'Max');
TileSize = get(handles.Table, 'data');
TileX = TileSize(1,1);
TileY = TileSize(2,1);
TileZ = TileSize(3,1);
TileC = TileSize(4,1);
z = get(handles.TileNumber,'Value');
c = get(handles.ColorNumber,'Value');
dataStitchXY = [];

for y = 1:TileY
    dataStitchX = [];
    for x = 1:TileX
        N = x + (y-1)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
        if Flag==1
            dataPart = handles.TileData{N}(:,:,pos);
        else
            dataPart = uint16(handles.TileHandle{N}(:,:,pos-1));
        end
        dataStitchX = cat(2, dataStitchX, dataPart);
    end
    dataStitchXY = cat(1, dataStitchXY, dataStitchX);
end
   
set(handles.imthandle, 'CData', dataStitchXY);
Current_title = sprintf('Before Merge: Slice-# %d', pos);
set(handles.hfig, 'Name', Current_title);
guidata(hObject, handles);

% hObject    handle to CurrentSlice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of CurrentSlice as text
%        str2double(get(hObject,'String')) returns contents of CurrentSlice as a double


% --- Executes during object creation, after setting all properties.
function CurrentSlice_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CurrentSlice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in AlignTiles.
function AlignTiles_Callback(hObject, eventdata, handles)
% hObject    handle to AlignTiles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
tic
guidata(hObject, handles);
SaveFlag = get(handles.SaveOverlap,'Value');
LoadFlag = get(handles.LoadDataMemory,'Value');
%AlignMethod = get(handles.AlignMethod,'Value');
FileName = get(handles.FileInfo,'String');

[dll_file, dll_path] = uigetfile('C:\Yicong_Program\CUDA_DLL\spimfusion_DLL\libapi_0605','select .h file');
dll_path = [dll_path, dll_file(1:end-2)];


TileSize = get(handles.Table, 'data');
TileX = TileSize(1,1);
TileY = TileSize(2,1);
TileZ = TileSize(3,1);
TileC = TileSize(4,1);

OverlapX = TileSize(1,2);
OverlapY = TileSize(2,2);
OverlapZ = TileSize(3,2);

ResX = TileSize(1,3);
ResY = TileSize(2,3);
ResZ = TileSize(3,3);

mRes = min(min(ResX, ResY), ResZ); 

Down = 2;
DownX = max(mRes/ResX*Down, 1);
DownY = max(mRes/ResY*Down, 1);
DownZ = max(mRes/ResZ*Down, 1); 
         
c = get(handles.ColorNumber,'Value'); 
k = ((TileY+2) * TileZ) * (c-1);
        for z=1:TileZ  
             k=k+2;
            for y = 1:TileY   
                k = k +1;
                FileNameX = FileName{k};
                TileName = strsplit(FileNameX,char(9));
                for x = 1:TileX
                    N = x + (y-1)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
                    handles.TileHandle{N} = TIFFStack([handles.path, TileName{x}], false, [], false);
                end          
            end  
        end          
        
for TileN = 1: TileX*TileY*TileZ
    [x, y, z] = ind2sub([TileX TileY TileZ], TileN);
    [ny, nx, nz] = size(handles.TileHandle{TileN});  % each tile only need to find the left and top tiles.  
    
    if x==1 && y==1
        handles.sx(x,y) = 1;
        handles.sy(x,y) = 1;
        handles.sz(x,y) = 1;
        handles.px(x,y) = 0; handles.py(x,y) = 0; handles.pz(x,y) = 0;
        handles.pxs(x,y) = 0; handles.pys(x,y) = 0; handles.pzs(x,y) = 0;  %sub-pixel shift
        overlap_top{1,1} = [0, 0];
        overlap_left{1,1} = [0, 0];
    end
    
%     if x==1 
%         overlap_left{x,y} = [0, 0];
%     end
%     
%     if y==1
%         overlap_top{x,y} = [0, 0];
%     end
    
    if y>=2
        N1 = x + (y-1)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
        N2 = x + (y-2)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
        handles.vertical_overlap_y(N1);
        y_crop = round(ny*handles.vertical_overlap_y(N1));
        Source = single(handles.TileHandle{N1}(1:y_crop,1:DownX:end,1:DownZ:end));  % Top of this Tile
        Target = single(handles.TileHandle{N2}(ny-y_crop+1:end,1:DownX:end,1:DownZ:end)); % bottom of the top Tile   
        
       % [M1, Overlap, NCC, Reg1] = Phasor(Source, Target, 1, 1);        
        [M, NCC, Reg] = RigidRegistration(Source, Target, dll_path);        
        handles.FileInfo.String{end+2} = ['Correlation coefficient between Tile_', 'Y', num2str(y-1), '_X', num2str(x), ' and  Tile_Y', num2str(y), '_X', num2str(x), ' is ', num2str(NCC)];
        drawnow();
        
        if SaveFlag == 1
            WriteTifStack(Target,[handles.path,'Overlap_Bottom_', 'TileY', num2str(y-1), '_TileX', num2str(x),'.tif'],'32'); 
            WriteTifStack(Source,[handles.path,'Overlap_Top_BeforeReg_', 'TileY', num2str(y), '_TileX', num2str(x),'.tif'],'32'); 
            WriteTifStack(Reg,[handles.path,'Overlap_Top_AfterReg_', 'TileY', num2str(y), '_TileX', num2str(x),'.tif'],'32'); 
        end
        
        M_y(1) = M(1) + y_crop; % shift value from y dimension 
        M_y(2) = M(2)*DownX;
        M_y(3) = M(3)*DownZ;
                
        handles.py(x, y) = floor(M_y(1));  % p is phase shift in x, y, z. compare y vs. x. in intergral ; move up is positive; move left is positive
        handles.px(x, y) = floor(M_y(2));
        handles.pz(x, y) = floor(M_y(3));
        handles.pys(x, y) = M_y(1) - handles.py(x, y);  % ps is sub-pixel shift...
        handles.pxs(x, y) = M_y(2) - handles.px(x, y); 
        handles.pzs(x, y) = M_y(3) - handles.pz(x, y); 
        if x==1
            handles.sx(x, y) = -handles.px(x, y) + handles.sx(x, y-1); % new start position of the tile in the merged image based on Y shift
        end
        handles.sy(x, y) = -handles.py(x, y) + handles.sy(x, y-1) + ny;
        handles.sz(x, y) = -handles.pz(x, y) + handles.sz(x, y-1); 

        %overlap_top{x,y} = floor([M_y(1), M_y(2)]);
    end                 
    if x>=2
        N1 = x + (y-1)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
        N2 = (x-1) + (y-1)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
        x_crop = round(ny*handles.horizontal_overlap_x(N1));
        Source = single(handles.TileHandle{N1}(1:DownY:end,1:x_crop,1:DownZ:end)); % Left of this Tile
        Target = single(handles.TileHandle{N2}(1:DownY:end,nx-x_crop+1:end,1:DownZ:end)); % right of the Left Tile
        
        [M, NCC, Reg] = RigidRegistration(Source, Target, dll_path);   
        
        handles.FileInfo.String{end+2} = ['Correlation coefficient between Tile_', 'Y', num2str(y), '_X', num2str(x-1), ' and  Tile_Y', num2str(y), '_X', num2str(x), ' is ', num2str(NCC)];
        drawnow();      
        if SaveFlag == 1
            WriteTifStack(Target,[handles.path,'Overlap_Right_', 'TileY', num2str(y), '_TileX', num2str(x-1),'.tif'],'32'); 
            WriteTifStack(Reg,[handles.path,'Overlap_Left_AfterReg_', 'TileY', num2str(y), '_TileX', num2str(x),'.tif'],'32'); 
            WriteTifStack(Source,[handles.path,'Overlap_Left_BeforeReg_', 'TileY', num2str(y), '_TileX', num2str(x),'.tif'],'32'); 
        end
        
        M_x(1) = M(1)*DownY;
        M_x(2) = M(2) + x_crop;
        M_x(3) = M(3)*DownZ;    
        handles.py(x, y) = floor(M_x(1));  % p is phase shift in x, y, z. compare y vs. x. in intergral ; move up is positive; move left is positive
        handles.px(x, y) = floor(M_x(2));
        handles.pz(x, y) = floor(M_x(3));
        handles.pys(x, y) = M_x(1) - handles.py(x, y); % ps is sub-pixel shift...
        handles.pxs(x, y) = M_x(2) - handles.px(x, y); 
        handles.pzs(x, y) = M_x(3) - handles.pz(x, y);
        handles.sx(x, y) = -handles.px(x, y) + handles.sx(x-1, y) + nx; % new start position of the tile in the merged image based on X Shift
        if y==1
            handles.sy(x, y) = -handles.py(x, y) + handles.sy(x-1, y);
        end
        handles.sz(x, y) = -handles.pz(x, y) + handles.sz(x-1, y);
       % overlap_left{x,y} = floor([M_x(1), M_x(2)]);   

    end         
     
    handles.ex(x, y) = handles.sx(x, y) + nx -1;  % new end position of the tile in the merged image
    handles.ey(x, y) = handles.sy(x, y) + ny -1;
    handles.ez(x, y) = handles.sz(x, y) + nz -1;   
    
end

msx = min(handles.sx(:)); msy = min(handles.sy(:)); msz = min(handles.sz(:));
handles.sx = handles.sx - msx + 1;  % restet from 1 to end. 
handles.sy = handles.sy - msy + 1;
handles.sz = handles.sz - msz + 1;
handles.ex = handles.ex - msx + 1;
handles.ey = handles.ey - msy + 1;
handles.ez = handles.ez - msz + 1;

for TileN = 2: TileX*TileY*TileZ
    [x, y, z] = ind2sub([TileX TileY TileZ], TileN);
    if y>=2
        overlap_top{x,y} = [handles.ey(x,y-1) - handles.sy(x,y) + 1, handles.sx(x,y) - handles.sx(x,y-1)];  % y, x 
    else
        overlap_top{x,y} = [0, 0];
    end
    
    if x>=2
        overlap_left{x,y} = [handles.sy(x,y) - handles.sy(x-1,y), handles.ex(x-1,y) - handles.sx(x,y) + 1];  % y,  
    else
         overlap_left{x,y} = [0, 0];
    end
end

% calculate the weigthed image - x,y for each tile 
for TileN = 1: TileX*TileY*TileZ
    [x, y, z] = ind2sub([TileX TileY TileZ], TileN);
    if y>=2
        overlap_top{x,y} = [handles.ey(x,y-1) - handles.sy(x,y) + 1, handles.sx(x,y) - handles.sx(x,y-1)];  % y, x 
    else
        overlap_top{x,y} = [0, 0];
    end
    
    if x>=2
        overlap_left{x,y} = [handles.sy(x,y) - handles.sy(x-1,y),handles.ex(x-1,y) - handles.sx(x,y) + 1];  % y,  
    else
        overlap_left{x,y} = [0, 0];
    end
end
   
% calculate regions to crop.
for TileN = 1: TileX*TileY*TileZ
    [x, y, z] = ind2sub([TileX TileY TileZ], TileN);  
    if x+1>TileX 
        overlap_left{x+1,y} = [0, 0];
    end
    
    if y+1>TileY 
        overlap_top{x,y+1} = [0, 0];
    end
    
    
    if overlap_left{x,y}(1) > 0   %negative, then set top zeros; positive then bottom zeros
        pad_bottom1 = overlap_left{x,y}(1);
        pad_top1 = 0;
    elseif overlap_left{x,y}(1) < 0
        pad_top1 = -overlap_left{x,y}(1);
        pad_bottom1 = 0;
    else
        pad_bottom1 = 0;
        pad_top1 = 0;
    end

    if overlap_left{x+1,y}(1) > 0   %negative, then set top zeros; positive then bottom zeros
        pad_top2 = overlap_left{x+1,y}(1);  
        pad_bottom2 = 0;
    elseif overlap_left{x+1,y}(1) < 0
        pad_bottom2 = -overlap_left{x+1,y}(1); 
        pad_top2 = 0;
    else
        pad_bottom2 = 0;
        pad_top2 = 0;
    end

    if overlap_top{x,y}(2) > 0  % > 0; set zeros on the right, else < 0 set zeros on the left
        pad_right1 = overlap_top{x,y}(2);
        pad_left1 = 0;
    elseif overlap_top{x,y}(2) < 0
        pad_left1 = -overlap_top{x,y}(2);
        pad_right1 = 0;
    else 
        pad_right1 = 0;
        pad_left1 = 0;
    end

    if overlap_top{x,y+1}(2) > 0    % > 0; set zeros on the right, else < 0 set zeros on the left 
      pad_left2 = overlap_top{x,y+1}(2);
      pad_right2 = 0;
    elseif overlap_top{x,y+1}(2) < 0
       pad_right2 = -overlap_top{x,y+1}(2);
       pad_left2 = 0;
    else
       pad_right2 = 0;
       pad_left2 = 0;
    end

    crop_bottom{x,y} = max(pad_bottom1,pad_bottom2);
    crop_top{x,y} = max(pad_top1,pad_top2);
    crop_left{x,y} = max(pad_left1,pad_left2);
    crop_right{x,y} = max(pad_right1,pad_right2);
end

for TileN = 1: TileX*TileY*TileZ
     [x, y, z] = ind2sub([TileX TileY TileZ], TileN);   
    if x<=TileX-1
        overlap_right{x,y} = overlap_left{x+1,y}(2) - crop_right{x,y} - crop_left{x+1,y};
    else
        overlap_right{x,y} = 0;
        crop_right{x,y} = 0;
    end
    
    if y<=TileY-1
        overlap_bottom{x,y} = overlap_top{x,y+1}(1) - crop_bottom{x,y} - crop_top{x,y+1};
    else
        overlap_bottom{x,y} = 0;
        crop_bottom{x,y} = 0;
    end
    
    if x>=2
        overlap_left{x,y} = overlap_left{x,y}(2) - crop_right{x-1,y} - crop_left{x,y};
    else
        overlap_left{x,y} = 0;
        crop_left{x,y} = 0;
    end  
    
    if y>=2
        overlap_top{x,y} = overlap_top{x,y}(1) - crop_top{x,y} - crop_bottom{x,y-1};
    else
        overlap_top{x,y} = 0;
        crop_top{x,y} = 0;
    end
       
    handles.W{x,y} = Weigth([ny, nx], [crop_top{x,y}, overlap_top{x,y}], [crop_bottom{x,y}, overlap_bottom{x,y}], [crop_left{x,y}, overlap_left{x,y}], [crop_right{x,y}, overlap_right{x,y}]);
              
end

handles.ny_merge = max(handles.ey(:));
handles.nx_merge = max(handles.ex(:));
handles.nz_merge = max(handles.ez(:));

% WriteTifStack(handles.W{1,1},'k:\WeightImage1.tif','32');
% WriteTifStack(handles.W{1,2},'k:\WeightImage2.tif','32');

handles.FileInfo.String{end+1} = ['Translation factors between the tiles have been calculated, check the merge results ... .'];

t = toc;
handles.FileInfo.String{end+1} = ['This step takes ', num2str(floor(t)), ' seconds!'];
guidata(hObject, handles);




% for y = 1:TileY
%     dataPart = uint16(handles.TileData{y,1,1}(1:scaleXY:end,1:scaleXY:end,1:scaleZ:end));
%     TileName = char(FileName{z}(y,x));
%     dataStitch = cat(1, dataPart, dataStitch);
% end


% --- Executes on slider movement.
function posSlider_MergeData_Callback(hObject, eventdata, handles)
% hObject    handle to posSlider_MergeData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
guidata(hObject, handles);
imin = get(handles.posSlider_MergeData,'Min');
imax = get(handles.posSlider_MergeData,'Max');
pos = round(get(handles.posSlider_MergeData,'Value'));
set(handles.CurrentSliceMerge, 'String', ['z:' num2str(pos)]);

MarkFlag = get(handles.MarkOverlap,'Value');

TileSize = get(handles.Table, 'data');

TileX = TileSize(1,1);
TileY = TileSize(2,1);
TileZ = TileSize(3,1);

ny = handles.ny_merge;
nx = handles.nx_merge;
nz = handles.nz_merge;

TileMerge = zeros(handles.ny_merge,handles.nx_merge);   
Overlap = zeros(handles.ny_merge,handles.nx_merge);  

for TileN = 1: TileX*TileY*TileZ
        [x, y, z] = ind2sub([TileX TileY TileZ], TileN);          
        z1 = pos + handles.pz(x,y);
        z2 = z1 + 1;
        Slices = single(handles.TileHandle{TileN}(:,:,z1:z2));
        SliceXY = Slices(:,:,1)*(1-handles.pzs(y)) + Slices(:,:,2) * handles.pzs(y);  % for subpixel shift in z
        Tile{x, y} = imtranslate(SliceXY,[-handles.pxs(x, y), -handles.pys(x, y)]);  % shift in xy with subpixel; make sure it is positive        
        TileMerge(handles.sy(x,y):handles.ey(x,y),handles.sx(x,y):handles.ex(x,y)) = single(TileMerge(handles.sy(x,y):handles.ey(x,y),handles.sx(x,y):handles.ex(x,y))) + Tile{x,y}.*handles.W{x,y}; % top lef is positive     
%         if MarkFlag==1
%             Overlap(handles.sy(x,y):handles.ey(x,y),handles.sx(x,y):handles.ex(x,y)) = Overlap(handles.sy(x,y):handles.ey(x,y),handles.sx(x,y):handles.ex(x,y)) + 200*abs(handles.W{x,y}-1);
%             handles.TileMerge = imfuse(handles.TileMerge,Overlap);
%         else
            handles.TileMerge = TileMerge;
%         end
end

im_max = max(handles.TileMerge(:));
im_min = min(handles.TileMerge(:));
%handles.hfigMerge = imtool(handles.TileMerge,[im_min im_max]);
handles.imthandleMerge = findobj(handles.hfigMerge,'type','image');
set(handles.imthandleMerge, 'CData', handles.TileMerge);

Current_title = sprintf('After Merge: Slice-# %d', pos);
set(handles.hfigMerge, 'Name', Current_title);

    


   
guidata(hObject, handles);


% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function posSlider_MergeData_CreateFcn(hObject, eventdata, handles)
% hObject    handle to posSlider_MergeData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function CurrentSliceMerge_Callback(hObject, eventdata, handles)
% hObject    handle to CurrentSliceMerge (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of CurrentSliceMerge as text
%        str2double(get(hObject,'String')) returns contents of CurrentSliceMerge as a double


% --- Executes during object creation, after setting all properties.
function CurrentSliceMerge_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CurrentSliceMerge (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in OrganizeTiles.
function OrganizeTiles_Callback(hObject, eventdata, handles)
% hObject    handle to OrganizeTiles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
guidata(hObject, handles);
FileNumber = length(handles.FileName);

TileSize = get(handles.Table, 'data');

TileX = TileSize(1,1);
TileY = TileSize(2,1);
TileZ = TileSize(3,1);
TileC = TileSize(4,1);

if TileX * TileY * TileZ * TileC == FileNumber
    handles.FileInfo.String = ['There are ', num2str(FileNumber), ' files, check if it matches the tile format!'];
else 
    handles.FileInfo.String = ['There are ', num2str(FileNumber), ' files, but it does not match the tile size. Correct it'];
end

 idx1 = get(handles.TileOrder1,'Value'); items1 = get(handles.TileOrder1,'String');  % normal or reverse
 idx2 = get(handles.TileOrder2,'Value'); items2 = get(handles.TileOrder2,'String');  % order for x, y, z, c
 
 if idx1 == 1
    FileName = handles.FileName;
 else
    FileName = flip(handles.FileName);
 end
 TileOrder = reshape(FileName, TileX, TileY, TileZ, TileC);
 
  switch items2{idx2}
     case 'X Y Z C'
        
     case 'Y X Z C'
        TileOrder = permute(TileOrder,[2 1 3 4]);  
      
     case 'X Y C Z'
        TileOrder = permute(TileOrder,[1 2 4 3]);
        
     case 'Y X C Z'
        TileOrder = permute(TileOrder,[2 1 4 3]);  
        
     case 'C X Y Z'
        TileOrder = permute(TileOrder,[4 1 2 3]);
        
     case 'C Y X Z'
        TileOrder = permute(TileOrder,[4 2 1 3]);
               
  end
 
   N = 1;
   k = 0;
   for c=1:TileC
    for z=1:TileZ     
      k=k+2;
      final{k} = ['XY Tiles at TileZ = ', num2str(z), '; TileC = ', num2str(c)];
     for y=1:TileY
         k = k + 1;
         filelist = '';
        for x=1:TileX
            fn = [char(TileOrder(N)), char(9)];
            filelist = [filelist, fn];
            N = N + 1;
        end
        final{k} = filelist; 
     end
    end
   end
  
set(handles.FileInfo,'String', final);
 
OverlapX = TileSize(1,2);
OverlapY = TileSize(2,2);
OverlapZ = TileSize(3,2);

for N=1:FileNumber
    handles.vertical_overlap_y(N) = OverlapY;
    handles.horizontal_overlap_y(N) = 1;
    handles.horizontal_overlap_x(N) = OverlapX;
    handles.vertical_overlap_x(N) = 1;
    handles.horizontal_overlap_z(N) = OverlapZ;
    handles.vertical_overlap_z(N) = OverlapZ;
end
 
for z = 1:TileZ
    listZ{z} = num2str(z);
end
set(handles.TileNumber, 'String', listZ);

for c = 1:TileC
    listC{c} = num2str(c);
end
set(handles.ColorNumber, 'String', listC);

guidata(hObject, handles);
   
  
  
  
function Path_Callback(hObject, eventdata, handles)
% hObject    handle to Path (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Path as text
%        str2double(get(hObject,'String')) returns contents of Path as a double


% --- Executes during object creation, after setting all properties.
function Path_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Path (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton1


% --- Executes on button press in checkbox2.
function checkbox2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox2


% --- Executes on selection change in TileOrder1.
function TileOrder1_Callback(hObject, eventdata, handles)
% hObject    handle to TileOrder1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns TileOrder1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from TileOrder1


% --- Executes during object creation, after setting all properties.
function TileOrder1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TileOrder1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in TileOrder2.
function TileOrder2_Callback(hObject, eventdata, handles)
% hObject    handle to TileOrder2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns TileOrder2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from TileOrder2


% --- Executes during object creation, after setting all properties.
function TileOrder2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TileOrder2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in AutoCalculationTileOrders.
function AutoCalculationTileOrders_Callback(hObject, eventdata, handles)
% hObject    handle to AutoCalculationTileOrders (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
tic
guidata(hObject, handles);
GPU_Flag = get(handles.GPU,'Value');
LoadFlag = get(handles.LoadDataMemory,'Value');

TileSize = get(handles.Table, 'data');
TileX = TileSize(1,1);
TileY = TileSize(2,1);
TileZ = TileSize(3,1);
TileC = TileSize(4,1);

FileName = get(handles.FileInfo,'String');
zoom = 1/max(handles.scaleX, handles.scaleY);

c = get(handles.ColorNumber,'Value'); 
k = ((TileY+2) * TileZ) * (c-1);
        for z=1:TileZ  
             k=k+2;
            for y = 1:TileY   
                k = k +1;
                FileNameX = FileName{k};
                TileName = strsplit(FileNameX,char(9));
                for x = 1:TileX
                        N = x + (y-1)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
                    if LoadFlag == 0
                        handles.TileHandle{N} = TIFFStack([handles.path, TileName{x}], false, [], false);
                        [ny, nx, nz] = size(handles.TileHandle{N});                        
                        z_range = 1:handles.scaleZ:nz; 
                        TileDown{N} = ReadBigTifStack([handles.path, TileName{x}], z_range, zoom);
                    elseif LoadFlag == 1
                        TileDown{N} = handles.TileData{N}(1:handles.scaleY:end,1:handles.scaleX:end,1:handles.scaleZ:end);
                    end
                end          
            end  
        end

  
% calucate the shift of each tile based on the channel selected. 
%c = get(handles.ColorNumber,'Value'); 
for TileN = 1: TileX*TileY*TileZ
    [x, y, z] = ind2sub([TileX TileY TileZ], TileN);
    handles.vertical_overlap_y(TileN) = 1;
    handles.horizontal_overlap_x(TileN) = 1;
    handles.vertical_overlap_z(TileN) = 1;
    handles.horizontal_overlap_z(TileN) = 1;

    if y>=2
            N1 = x + (y-1)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
            N2 = x + (y-2)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
%             WriteTifStack(TileDown{N1},'K:\StitcherTest\t1.tif','32');
%             WriteTifStack(TileDown{N2},'K:\StitcherTest\t2.tif','32');
           [M, Overlap, NCC] = Phasor(TileDown{N1}, TileDown{N2},GPU_Flag, 0);  
           handles.vertical_overlap_y(TileN) = Overlap(1);
           handles.vertical_overlap_x(TileN) = Overlap(2);
           handles.vertical_overlap_z(TileN) = Overlap(3);
           handles.vertical_correlation(TileN) = NCC;
           handles.FileInfo.String{end+2} = ['Overlap between Tile_', 'Y', num2str(y-1), '_X', num2str(x), ' and Tile_Y', num2str(y), '_X', num2str(x), ' is ', num2str(handles.vertical_overlap_y(TileN))];
           drawnow();        
    end
    
    if x>=2
            N1 = x + (y-1)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
            N2 = (x-1) + (y-1)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
            [M, Overlap, NCC] = Phasor(TileDown{N1}, TileDown{N2}, GPU_Flag, 0);
            handles.horizontal_overlap_y(TileN) = Overlap(1);
            handles.horizontal_overlap_x(TileN) = Overlap(2);
            handles.horizontal_overlap_z(TileN) = Overlap(3);
            handles.horizontal_correlation(TileN) = NCC;
            handles.FileInfo.String{end+2} = ['Overlap between Tile_', 'Y', num2str(y), '_X', num2str(x-1), ' and Tile_Y', num2str(y), '_X', num2str(x), ' is ', num2str(handles.horizontal_overlap_x(TileN))]; 
    end
 end
       
TileSize(2,2)= min(handles.vertical_overlap_y);
TileSize(1,2) = min(handles.horizontal_overlap_x);
TileSize(3,2)= min(min(handles.vertical_overlap_z),min(handles.horizontal_overlap_z));
set(handles.Table, 'data', TileSize);
t = toc;
handles.FileInfo.String{end+1} = ['This step takes ', num2str(floor(t)), ' seconds!'];
guidata(hObject, handles);
     



% --- Executes on button press in PhaseShift.
function PhaseShift_Callback(hObject, eventdata, handles)
% hObject    handle to PhaseShift (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of PhaseShift


% --- Executes on button press in Registration.
function Registration_Callback(hObject, eventdata, handles)
% hObject    handle to Registration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Registration


% --- Executes on button press in ViewAfterStitching.
function ViewAfterStitching_Callback(hObject, eventdata, handles)
% hObject    handle to ViewAfterStitching (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

TileSize = get(handles.Table, 'data');
MarkFlag = get(handles.MarkOverlap,'Value');

TileX = TileSize(1,1);
TileY = TileSize(2,1);
TileZ = TileSize(3,1);
TileC = TileSize(4,1);
c = get(handles.ColorNumber,'Value'); 

ny = handles.ny_merge; 
nx = handles.nx_merge; 
nz = handles.nz_merge;

set(handles.posSlider_MergeData,'Min',1);
set(handles.posSlider_MergeData,'Max',nz);
pos = round(nz/2);
set(handles.CurrentSliceMerge, 'String', ['z:' num2str(pos)]);
set(handles.posSlider_MergeData,'SliderStep',[1/(nz-1), 0.01]);
set(handles.posSlider_MergeData,'Value',pos);

TileMerge = zeros(handles.ny_merge,handles.nx_merge);   
Overlap = zeros(handles.ny_merge,handles.nx_merge);  


for TileN = 1: TileX*TileY*TileZ
        [x, y, z] = ind2sub([TileX TileY TileZ], TileN);
%       z1 = pos + handles.pz(x,y);
        z1 = pos;
        z2 = z1 + 1;
        Slices = single(handles.TileHandle{TileN+(c-1)*TileX*TileY*TileZ}(:,:,z1:z2));
        SliceXY = Slices(:,:,1)*(1-handles.pzs(y)) + Slices(:,:,2) * handles.pzs(y);  % for subpixel shift in z
        Tile{x, y} = imtranslate(SliceXY,[-handles.pxs(x, y), -handles.pys(x, y)]);  % shift in xy with subpixel; make sure it is positive        
        TileMerge(handles.sy(x,y):handles.ey(x,y),handles.sx(x,y):handles.ex(x,y)) = single(TileMerge(handles.sy(x,y):handles.ey(x,y),handles.sx(x,y):handles.ex(x,y))) + Tile{x,y}.*handles.W{x,y}; % top lef is positive     
        if MarkFlag==1
            Overlap(handles.sy(x,y):handles.ey(x,y),handles.sx(x,y):handles.ex(x,y)) = Overlap(handles.sy(x,y):handles.ey(x,y),handles.sx(x,y):handles.ex(x,y)) + 250*abs(handles.W{x,y}-1);
        else
            handles.TileMerge = TileMerge;
        end
end
      

if MarkFlag==1
    handles.hfigMerge = imtool(imfuse(handles.TileMerge, Overlap, 'falsecolor','Scaling','joint','ColorChannels',[1 2 0]));;
else
    im_max = max(handles.TileMerge(:));
    im_min = min(handles.TileMerge(:));
    handles.hfigMerge = imtool(handles.TileMerge,[im_min im_max]);
end
    
handles.imthandleMerge = findobj(handles.hfigMerge,'type','image');

Current_title = sprintf('After Merge: Slice-# %d', pos);
set(handles.hfigMerge, 'Name', Current_title);
    
guidata(hObject, handles);


% --- Executes on button press in SaveFiles.
function SaveFiles_Callback(hObject, eventdata, handles)


% hObject    handle to SaveFiles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

guidata(hObject, handles);

if get(handles.SaveStack,'value') == 1
    SavingMode = 1; % save as stack
elseif get(handles.SaveSlice,'value') == 1
    SavingMode = 2; % each slice save as a file
end

saving_path  = handles.OutputPath.String;
[saving_file,saving_path] = uiputfile([saving_path, '*.tif'], 'File Selection');
set(handles.OutputPath, 'String', saving_path);

TileSize = get(handles.Table, 'data');
TileX = TileSize(1,1);
TileY = TileSize(2,1);
TileZ = TileSize(3,1);
TileC = TileSize(4,1);

ny = handles.ny_merge;
nx = handles.nx_merge;
nz = handles.nz_merge;

% for cropping
z_range = max(handles.sz(:)): min(handles.ez(:));
y_range = max(handles.sy(:,1)): min(handles.ey(:,end));
x_range = max(handles.sx(1,:)): min(handles.ex(end,:));


% loading all Tiles Handling
FileName = get(handles.FileInfo,'String');
k = 0;
N = 1;  
for c=1:TileC
    for z=1:TileZ     
         k=k+2;
        for y = 1:TileY   
            k = k + 1;
            FileNameX = FileName{k};
            TileName = strsplit(FileNameX,char(9));
            for x = 1:TileX
               handles.TileHandle{N} = TIFFStack([handles.path, TileName{x}], false, [], true);
               N = N + 1;
            end
        end
    end
end

if SavingMode == 1  % Stack

 for c = 1:TileC
     tic
     outTiff = Tiff([saving_path,saving_file(1:end-4),'_C', num2str(c), '.tif'],'w8');
   for k=1:length(z_range)
        disp(['fusing and saving channel: ',num2str(c), '; slice: ', num2str(k)]);
        TileMerge = zeros(handles.ny_merge,handles.nx_merge);   
        for TileN = 1: TileX*TileY*TileZ
            [x, y, z] = ind2sub([TileX TileY TileZ], TileN);      
            z1 = k + handles.pz(x,y);
            z2 = z1 + 1;
            [ny1, nx1, nz1] = size(handles.TileHandle{TileN + (c-1)*TileX*TileY*TileZ});
             if z1>=1 & z2<=nz1
                 Slices = single(handles.TileHandle{TileN +(c-1)*TileX*TileY*TileZ}(:,:,z1:z2));
                 SliceXY = Slices(:,:,1)*(1-handles.pzs(x,y)) + Slices(:,:,2) * handles.pzs(x,y);  % for subpixel shift in z
                 SliceXY = imtranslate(SliceXY,[-handles.pxs(x, y), -handles.pys(x, y)]);  % shift in xy with subpixel; make sure it is positive  
             else
                 SliceXY = zeros(ny1,nx1);
             end
            TileMerge(handles.sy(x,y):handles.ey(x,y),handles.sx(x,y):handles.ex(x,y)) = TileMerge(handles.sy(x,y):handles.ey(x,y),handles.sx(x,y):handles.ex(x,y)) + SliceXY.*handles.W{x,y}; % top lef is positive     
            %WriteTifStack(single(TileMerge),['K:\StitcherTest\Gut_down\WeightImage\TileMerge', num2str(TileN), '.tif'],'32');
        end    

        WriteBigStack(TileMerge(y_range,x_range), outTiff, '16', k);
        if mod(k, round(length(z_range)/10)) == 0
            saving_percentage = floor((k-1)/length(z_range)* 100);
            handles.FileInfo.String{end} = ['Stitching Channel', num2str(c), ' is Saving ', num2str(saving_percentage), '%'];
            drawnow();
        end
   end
    
    outTiff.close();
    handles.FileInfo.String{end+1} = ['Stitching Channel', num2str(c), ' is Done and Saved !'];
    t = toc;
    handles.FileInfo.String{end+1} = ['This step takes ', num2str(floor(t)), ' seconds!'];
    drawnow(); 
 end
     
elseif SavingMode == 2  % single slices   
  for c = 1:TileC 
      tic
    parfor k=1:length(z_range)
        disp(['fusing and saving channel: ',num2str(c), '; slice: ', num2str(k)]);
        TileMerge = zeros(handles.ny_merge,handles.nx_merge);   
        for TileN = 1:TileX*TileY*TileZ
             [x, y, z] = ind2sub([TileX TileY TileZ], TileN);     
             z1 = k + handles.pz(x,y);
             z2 = z1 + 1;
             [ny1, nx1, nz1] = size(handles.TileHandle{TileN + (c-1)*TileX*TileY*TileZ});
             if z1>=1 & z2<=nz1
                 Slices = single(handles.TileHandle{TileN +(c-1)*TileX*TileY*TileZ}(:,:,z1:z2));
                 SliceXY = Slices(:,:,1)*(1-handles.pzs(x,y)) + Slices(:,:,2) * handles.pzs(x,y);  % for subpixel shift in z
                 SliceXY = imtranslate(SliceXY,[-handles.pxs(x, y), -handles.pys(x, y)]);  % shift in xy with subpixel; make sure it is positive  
             else 
                 SliceXY = zeros(ny1,nx1);
             end   
             TileMerge(handles.sy(x,y):handles.ey(x,y),handles.sx(x,y):handles.ex(x,y)) = TileMerge(handles.sy(x,y):handles.ey(x,y),handles.sx(x,y):handles.ex(x,y)) + SliceXY.*handles.W{x,y}; % top lef is positive     
             %WriteTifStack(single(TileMerge),['K:\StitcherTest\Gut_down\WeightImage\TileMerge', num2str(TileN), '.tif'],'32');
        end        
        WriteTifStack(TileMerge(y_range,x_range),[saving_path, saving_file(1:end-4), '_C', num2str(c),'_Z', num2str(k), '.tif'], '16');
    end
    handles.FileInfo.String{end+1} = ['Stitching Channel', num2str(c), ' is Done and Saved !'];
    t = toc;
    handles.FileInfo.String{end+1} = ['This step takes ', num2str(floor(t)), ' seconds!'];
    drawnow(); 
  end
end


% --- Executes on button press in SaveStack.
function SaveStack_Callback(hObject, eventdata, handles)
% hObject    handle to SaveStack (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of SaveStack


% --- Executes on button press in GPU.
function GPU_Callback(hObject, eventdata, handles)
% hObject    handle to GPU (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of GPU


% --- Executes on button press in SaveSlice.
function SaveSlice_Callback(hObject, eventdata, handles)
% hObject    handle to SaveSlice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of SaveSlice



function OutputPath_Callback(hObject, eventdata, handles)
% hObject    handle to OutputPath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of OutputPath as text
%        str2double(get(hObject,'String')) returns contents of OutputPath as a double


% --- Executes during object creation, after setting all properties.
function OutputPath_CreateFcn(hObject, eventdata, handles)
% hObject    handle to OutputPath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in PhaseShiftCPU.
function PhaseShiftCPU_Callback(hObject, eventdata, handles)
% hObject    handle to PhaseShiftCPU (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of PhaseShiftCPU


% --- Executes on selection change in AlignMethod.
function AlignMethod_Callback(hObject, eventdata, handles)
% hObject    handle to AlignMethod (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns AlignMethod contents as cell array
%        contents{get(hObject,'Value')} returns selected item from AlignMethod


% --- Executes during object creation, after setting all properties.
function AlignMethod_CreateFcn(hObject, eventdata, handles)
% hObject    handle to AlignMethod (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in GPU.
function KnownTileFormat_Callback(hObject, eventdata, handles)
% hObject    handle to GPU (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of GPU


% --- Executes during object creation, after setting all properties.
function SelectFiles_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SelectFiles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in SaveOverlap.
function SaveOverlap_Callback(hObject, eventdata, handles)
% hObject    handle to SaveOverlap (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of SaveOverlap


% --- Executes on button press in MarkOverlap.
function MarkOverlap_Callback(hObject, eventdata, handles)
% hObject    handle to MarkOverlap (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of MarkOverlap


% --- Executes on button press in checkbox10.
function checkbox10_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox10


% --- Executes on button press in LoadDataMemory.
function LoadDataMemory_Callback(hObject, eventdata, handles)
% hObject    handle to LoadDataMemory (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LoadDataMemory


% --- Executes on selection change in TileNumber.
function TileNumber_Callback(hObject, eventdata, handles)
% hObject    handle to TileNumber (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
guidata(hObject, handles);
Flag = get(handles.LoadDataMemory,'Value');
imin = get(handles.posSlider_RawData,'Min');
imax = get(handles.posSlider_RawData,'Max');
pos = round(get(handles.posSlider_RawData,'Value'));
TileSize = get(handles.Table, 'data');
TileX = TileSize(1,1);
TileY = TileSize(2,1);
TileZ = TileSize(3,1);
TileC = TileSize(4,1);

z = get(handles.TileNumber,'Value');
c = get(handles.ColorNumber,'Value');
dataStitchXY = [];
for y = 1:TileY
    dataStitchX = [];
    for x = 1:TileX
        N = x + (y-1)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
        if Flag==1
            dataPart = handles.TileData{N}(:,:,pos);
        else
            dataPart = uint16(handles.TileHandle{N}(:,:,pos-1));
        end
        dataStitchX = cat(2, dataStitchX, dataPart);
    end
    dataStitchXY = cat(1, dataStitchXY, dataStitchX);
end

set(handles.imthandle, 'CData', dataStitchXY);
Current_title = sprintf('Before Merge: Slice-# %d', pos);
set(handles.hfig, 'Name', Current_title);
guidata(hObject, handles);

% Hints: contents = cellstr(get(hObject,'String')) returns TileNumber contents as cell array
%        contents{get(hObject,'Value')} returns selected item from TileNumber


% --- Executes during object creation, after setting all properties.
function TileNumber_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TileNumber (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ColorNumber.
function ColorNumber_Callback(hObject, eventdata, handles)
% hObject    handle to ColorNumber (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
guidata(hObject, handles);
Flag = get(handles.LoadDataMemory,'Value');
imin = get(handles.posSlider_RawData,'Min');
imax = get(handles.posSlider_RawData,'Max');
pos = round(get(handles.posSlider_RawData,'Value'));
TileSize = get(handles.Table, 'data');
TileX = TileSize(1,1);
TileY = TileSize(2,1);
TileZ = TileSize(3,1);
TileC = TileSize(4,1);

z = get(handles.TileNumber,'Value');
c = get(handles.ColorNumber,'Value');

dataStitchXY = [];

for y = 1:TileY
    dataStitchX = [];
    for x = 1:TileX
        N = x + (y-1)*TileX + (z-1)*TileX*TileY + (c-1)*TileX*TileY*TileZ;
        if Flag==1
            dataPart = handles.TileData{N}(:,:,pos);
        else
            dataPart = uint16(handles.TileHandle{N}(:,:,pos-1));
        end
        dataStitchX = cat(2, dataStitchX, dataPart);
    end
    dataStitchXY = cat(1, dataStitchXY, dataStitchX);
end
   
set(handles.imthandle, 'CData', dataStitchXY);
Current_title = sprintf('Before Merge: Slice-# %d', pos);
set(handles.hfig, 'Name', Current_title);
guidata(hObject, handles);

% Hints: contents = cellstr(get(hObject,'String')) returns ColorNumber contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ColorNumber

% --- Executes during object creation, after setting all properties.
function ColorNumber_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ColorNumber (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function ViewAfterStitching_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ViewAfterStitching (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
