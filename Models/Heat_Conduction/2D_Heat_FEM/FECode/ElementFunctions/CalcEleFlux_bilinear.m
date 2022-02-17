%%                          CalcEleFlux_bilinear
%
% Institut fuer Statik | TU Braunschweig
% Beethovenstrasse 51
% 38106 Braunschweig
%
% The routine CalcEleForce_bilinear is in charge of calculating the element
% heat flux vector Sigma for bilinear ansatz functions.
%
% Input data:       material:    	Conductivity values for x- and
%                                   y-direction (real)                       
%                   x_hat:          Node coordinates (real)
%                   nUnknEleFlux:   Number of unknowns per element (heat
%                                   fluxes)
%                   v:              Temperature values of the element
%                                   nodes (real)
%                               
% Output data:      Sigma:          Element heat flux vector (real)

function [ Sigma ] = CalcEleFlux_bilinear(material, x_hat, nUnknEleFlux, v)

    % - Corner points of the element
    %----------------------------------------------------------------------
    cp = [  -1,   1,   1,  -1  ;   % xi coordinate
            -1,  -1    1,   1  ];  % eta coordinate
    % - Generating the needed variables:
    %----------------------------------------------------------------------
    
    lambdax = material(1);
    lambday = material(2);
    
    E = [  lambdax,        0  ;
                      0,  lambday  ];

    % - Building the element heat flux vector (Section 7 in handout)
    %----------------------------------------------------------------------
    
    % Initializing the element heat flux vector Sigma:
    Sigma = zeros(nUnknEleFlux,1);
    
    % Counter  
    cc = 1;
    
    for ii = 1:4 
        
        xi = cp(1,ii);
        eta = cp(2,ii);

        % Ansatz functions    
        Omega = [ 0.25*(1-eta)*(1-xi) 0.25*(1+xi)*(1-eta) 0.25*(1+xi)*(1+eta) 0.25*(1-xi)*(1+eta)];

        % Derivatives of the ansatz functions (with respect to local coordinates)

        Omega_dXI = [ -0.25*(1-eta) +0.25*(1-eta) +0.25*(1+eta) -0.25*(1+eta)
                      -0.25*(1-xi)  -0.25*(1+xi)  +0.25*(1+xi)  +0.25*(1-xi)];

        % Calculating the Jacobi-Matrix (refer Section 5.1.2 in handout):
        J       = [ Omega_dXI * x_hat ];
        detJ    = det(J);
        invJ    = (1/detJ) * [ J(2,2) -J(1,2)
                              -J(2,1)  J(1,1)]; 
        

        % Transformation of derivative of ansatz function from local
        % coordinate system to global coordinate system - (Section 5.2.2 in handout)
        H = - (invJ)*Omega_dXI;
            
        % Calculating the element heat flux vector pEle (refer Section 9 in handout):
        Sigma(cc:cc+1,1) = E * H * v;
            
        cc = cc +2;        
        
    end
           
end