function dataB = shift3D(dataA,offset)

% move down is positive; move right is positive; move more z is postive 
[sy, sx, sz] = size(dataA);
dataB = zeros(sy, sx, sz);
 if offset(1)>0 & offset(1)<=sy & offset(2)>0 & offset(2)<=sx & offset(3)>0 & offset(3)<=sz % > > > a a a
          dataB(offset(1)+1:end, offset(2)+1:end, offset(3)+1:end) = dataA(1:end-offset(1), 1:end-offset(2),1:end-offset(3));    
          %RdataA(end-offset(1)+1:end, end-offset(2)+1:end,end-offset(3)+1:end)=[];  
          dataB
          
          %dataB(offset(1)+1:end, offset(2)+1:end, offset(3)+1:end) = dataA(1:end-offset(1), 1:end-offset(2),1:end-offset(3));   
          
          
    elseif offset(1)<=0 & offset(1)>-sy & offset(2)<=0 & offset(2)>-sx & offset(3)<=0 & offset(3)>-sz % < < < b b b         
          dataB = dataB_2(-offset(1)+1:end,-offset(2)+1:end,-offset(3)+1:end);
        
    elseif offset(1)>0 & offset(1)<=sy & offset(2)<=0 & offset(2)>-sx & offset(3)<=0 & offset(3)>-sz % > < < a b b
        shiftB = dataB_2(1:end-offset(1),-offset(2)+1:end, -offset(3)+1:end);
     
    elseif offset(1)<=0 & offset(1)>-sy & offset(2)>0 & offset(2)<=sx & offset(3)>0 & offset(3)<=sz % < > > b a a
        shiftB = dataB_2(-offset(1)+1:end,1:end-offset(2),1:end-offset(3));
       
     elseif offset(1)>0 & offset(1)<=sy & offset(2)>0 & offset(2)<=sx & offset(3)<=0 & offset(3)>-sz % > > < a a b
         shiftB = dataB_2(1:end-offset(1),1:end-offset(2),-offset(3)+1:end);
       
    elseif offset(1)<=0 & offset(1)>-sy & offset(2)>0 & offset(2)<=sx & offset(3)<=0 & offset(3)>-sz% < > < b a b
        shiftB = dataB_2(-offset(1)+1:end,1:end-offset(2),-offset(3)+1:end);
      
    elseif offset(1)>0 & offset(1)<=sy & offset(2)<=0 & offset(2)>-sx & offset(3)>0 & offset(3)<=sz  % > < > a b a
        shiftB = dataB_2(1:end-offset(1),-offset(2)+1:end,1:end-offset(3));
    
    elseif offset(1)<=0 & offset(1)>sy & offset(2)<=0 & offset(2)>-sx & offset(3)>0 & offset(3)<=sz % < < > b b a
        shiftB = dataB_2(-offset(1)+1:end, -offset(2)+1:end, 1:end-offset(3));
      
    end




