function [Illumination] = IlluminationPattern(FWHM, N, mode, pixel,angle)

% unit of pixel size is in um;so that FHWM values is in um too.

FWHM = FWHM/pixel; % um
lamda = 0.488/pixel; %um
w0 = FWHM/(sqrt(2*log(2)));
zr = pi*w0^2/lamda;

nx_img = N(2);
nz_img = N(3);
ny = N(1);

for x=1:nx_img
    for z= 1:nz_img
        x1 = x - round(nx_img/2);
        z1 = z - round(nz_img/2);
        wz = w0*sqrt(1+(x1/zr)^2);
        ExcitationXZ(x,z) = (w0/wz)^2 * exp(-2*z1^2/wz^2);
    end
end

ExcitationZX = permute(ExcitationXZ,[2,1]);

if mode == 0 %% cross excitation for stage scanning mode, symmetrical case; for the visulization from objective view.
    Illumination = permute(repmat(ExcitationXZ + ExcitationZX,[1, 1, ny]),[3,1,2]);
else if mode == 1   %% to get epi
        Illumination = permute(repmat(ExcitationZX,[1, 1, ny]),[3,1,2]);
    else if mode == 2 %% regular sheet mode; 
           Illumination = permute(repmat(ExcitationXZ,[1, 1, ny]),[3,1,2]);
        else if mode == 3 %% sheet mode rotation for the visulization from coverslip;
                ExcitationZX = imrotate(ExcitationXZ,angle,'bilinear','crop');
                ExcitationZX_1 = imrotate(ExcitationXZ,-angle,'bilinear' ,'crop');
                Illumination = permute(repmat(ExcitationZX + ExcitationZX_1,[1, 1, ny]),[3,1,2]);  
            end
        end
    end
end
end

