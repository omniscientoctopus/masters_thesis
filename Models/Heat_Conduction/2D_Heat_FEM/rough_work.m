Inquiry_points = [[0.05635083 0.02817542 0.13888889 0.06944444]
                 [0.25       0.02817542 0.22222222 0.06944444]
                 [0.44364917 0.02817542 0.13888889 0.06944444]
                 [0.05635083 0.125      0.13888889 0.11111111]
                 [0.25       0.125      0.22222222 0.11111111]
                 [0.44364917 0.125      0.13888889 0.11111111]
                 [0.05635083 0.22182458 0.13888889 0.06944444]
                 [0.25       0.22182458 0.22222222 0.06944444]
                 [0.44364917 0.22182458 0.13888889 0.06944444]]; 
             
number_inquiry_points = size(Inquiry_points,1);

temperature_at_inquiry_points = zeros(number_inquiry_points,1);

for i = 1:number_inquiry_points

inquiry_point = Inquiry_points(i,1:2);

disp(inquiry_point)

% temperature_at_inquiry_points(i) = temperature_at_qp(inquiry_point, solution, nEleX, nEleY);

end