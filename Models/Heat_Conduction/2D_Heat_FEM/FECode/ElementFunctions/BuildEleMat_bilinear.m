%%                          BuildEleMat_bilinear
%
% Institut fuer Statik | TU Braunschweig
% Beethovenstrasse 51
% 38106 Braunschweig
%
% The routine BuildEleMat_bilinear is in charge of building the element
% stiffness matrix KEle for bilinear ansatz functions by using numerical
% integration.
%
% Input data:       material:    	Conductivity values for x- and
%                                   y-direction (real)
%                   x_hat:          Node coordinates (real)
%                   nUnknEleTemp:   Number of unknown temperature values in
%                                   the element (integer) 
%
% Output data:      KEle:           Element stiffness matrix (real)

function [ KEle ] = BuildEleMat_bilinear( material, x_hat, nUnknEleTemp )
   
    % - Values for the gaussian integration using 2 points
    %----------------------------------------------------------------------
    
    % Gauss points
    gp      = [-(1/sqrt(3)),  (1/sqrt(3))];

    % Weights
    weight  = [1,  1];

    % - Generating the needed variables:
    %----------------------------------------------------------------------
    
    lambdax = material(1);
    lambday = material(2);
    
    E = [  lambdax,        0  ;
                 0,  lambday  ];
    
    % - Building the element stiffness matrix KEle:
    %----------------------------------------------------------------------
    KEle = zeros(nUnknEleTemp,nUnknEleTemp);

    for ii = 1:2 
        
        xi = gp(ii);
        
        for jj = 1:2
            
            eta = gp(jj);

            % Ansatz functions    
            Omega = [ 0.25*(1-eta)*(1-xi) 0.25*(1+xi)*(1-eta) 0.25*(1+xi)*(1+eta) 0.25*(1-xi)*(1+eta)];
        
            % Derivatives of the ansatz functions (with respect to local coordinate system)
               
            Omega_dXI = [ -0.25*(1-eta) +0.25*(1-eta) +0.25*(1+eta) -0.25*(1+eta)
                          -0.25*(1-xi)  -0.25*(1+xi)  +0.25*(1+xi)  +0.25*(1-xi)];   
            
            % Calculating the Jacobi-Matrix (refer Section 5.1.2 in handout):
            J       =  Omega_dXI * x_hat ; 
                
            detJ    = det(J);
            invJ    = (1/detJ) * [ J(2,2) -J(1,2)
                                  -J(2,1)  J(1,1)]; 
             
            
            % Transformation of derivative of ansatz function from local
            % coordinate system to global coordinate system - (Section 5.2.2 in handout)
            H = - (invJ)*Omega_dXI;
            
            % Calculating the element stiffness matrix KEle (Section 6.1 & 7.1 in handout):
            KEle = KEle + transpose(H)*E*H*detJ*weight(ii)*weight(jj);

        end
    end
end
            
        


