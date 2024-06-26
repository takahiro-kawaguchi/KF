function [x_new, P_new, K, likelihood] = kf_update_likelihood (x, P, y, u, C, D, R)

F = (R+C*P*C');
K = P*C'/F;
e = (y-C*x-D*u);
x_new = x + K*e;
P_new = P-K*C*P;
likelihood = (e'/F)*e + log(det(F));
end 
