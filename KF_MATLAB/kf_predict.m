function [x_new, P_new] = kf_predict(x, P, u, A, Bu, B, Q)
x_new = A*x;
if ~isempty(u)
   x_new = x_new + B*u; 
end
P_new = A*P*A' + B*Q*B';
end 
