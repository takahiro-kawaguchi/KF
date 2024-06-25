function [x_new, P_new, K] = kf_update (x, P, y, C, R)
K = P*C'/(R+C*P*C');
x_new = x + K*(y-C*x);
P_new = P-K*C*P;
end 
