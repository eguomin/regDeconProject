function varargout = slice_view(varargin)
% SLICE_VIEW M-file for slice_view.fig
%      SLICE_VIEW, by itself, creates a new SLICE_VIEW or raises the existing
%      singleton*.
%
%      H = SLICE_VIEW returns the handle to a new SLICE_VIEW or the handle to
%      the existing singleton*.
%
%      SLICE_VIEW('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SLICE_VIEW.M with the given input arguments.
%
%      SLICE_VIEW('Property','Value',...) creates a new SLICE_VIEW or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before slice_view_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to slice_view_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help slice_view

% Last Modified by GUIDE v2.5 14-Mar-2011 20:48:03

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @slice_view_OpeningFcn, ...
                   'gui_OutputFcn',  @slice_view_OutputFcn, ...
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


% --- Executes just before slice_view is made visible.
function slice_view_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to slice_view (see VARARGIN)

handles.data = varargin{1};
sn = size(handles.data,3);
set(handles.posSlider,'Min',1);
set(handles.posSlider,'Max',sn);
set(handles.posSlider,'Value',round(sn/2));
set(handles.posSlider,'SliderStep',[1/(sn-1), 0.01]);
handles.pos = round(sn/2);

slice = handles.data(:,:,handles.pos);
im_max = max(slice(:));
im_min = min(slice(:));
handles.hfig = imtool(slice,[im_min im_max]);
handles.imthandle = findobj(handles.hfig,'type','image');
Current_title = sprintf('Slice-# %d', handles.pos);
set(handles.hfig, 'Name', Current_title);

% Choose default command line output for slice_view
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes slice_view wait for user response (see UIRESUME)
% uiwait(handles.SliceViewer);


% --- Outputs from this function are returned to the command line.
function varargout = slice_view_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on slider movement.
function posSlider_Callback(hObject, eventdata, handles)
% hObject    handle to posSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

imin = get(handles.posSlider,'Min');
imax = get(handles.posSlider,'Max');
currPos = get(handles.posSlider,'Value');
if (imin <= currPos <= imax)
    handles.pos = round(currPos);
    set(handles.posSlider,'Value',handles.pos);

    if(handles.rot == 0)
        slice = handles.data(:,:,handles.pos);
    else
        slice = flipud(rot90(handles.data(:,:,handles.pos)));
    end
    set(handles.imthandle, 'CData', slice);
    Current_title = sprintf('Slice-# %d', handles.pos);
    set(handles.hfig, 'Name', Current_title);
end
guidata(hObject, handles);


% --------------------------------------------------------------------
function Close_Callback(hObject, eventdata, handles)
% hObject    handle to Close (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

close(handles.hfig);
handles.data = [];
handles.pos = 0;
set(handles.posSlider, 'Enable', 'off');
guidata(hObject, handles);
