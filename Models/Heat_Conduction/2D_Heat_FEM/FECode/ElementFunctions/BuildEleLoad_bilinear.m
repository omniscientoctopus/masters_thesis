%%                          BuildEleLoad_bilinear
%
% Institut fuer Statik | TU Braunschweig
% Beethovenstrasse 51
% 38106 Braunschweig
%
% The routine BuildEleLoad_bilinear is in charge of building the element
% load vector fEle for bilinear ansatz functions by using numerical
% integration. 
%
% Input data:       p:              Value of the element load
%                   x_hat:          Coordinates of the loaded element
% Output data:      fEle:           Element load vector (real)

function [ fEle ] = BuildEleLoad_bilinear( p , x_hat)

    % - Values for the gaussian integration using 2 points
    %----------------------------------------------------------------------
    
    % Gauss points
    gp      = [-(1/sqrt(3)),  (1/sqrt(3))];

    % Weights
    weight  = [1,  1];
        
    % - Building the load vector
    %---------------------------------------------------------------------
    fEle = zeros(4,1);

    for ii = 1:2
        
        xi = gp(ii);
        
        for jj = 1:2

            eta = gp(jj);

            % Ansatz functions    
            Omega = [ 0.25*(1-eta)*(1-xi) 0.25*(1+xi)*(1-eta) 0.25*(1+xi)*(1+eta) 0.25*(1-xi)*(1+eta)];
        
            % Derivatives of the ansatz functions (with respect to local coordinate system)
               
            Omega_dXI = [ -0.25*(1-eta) +0.25*(1-eta) +0.25*(1+eta) -0.25*(1+eta)
                          -0.25*(1-xi)  -0.25*(1+xi)  +0.25*(1+xi)  +0.25*(1-xi)];   
           
            
            % Calculating the Jacobi-Matrix:
            J       = [ Omega_dXI * x_hat ]; 
            detJ    = det(J);
            
            % Calculating the element load vector fEle (Section 6.2 & 7.2 in handout):
            fEle = fEle + transpose(Omega) * p * detJ * weight(ii)*weight(jj);
            
        end
            
    end
    
end
            
            
            
            
                      
           

