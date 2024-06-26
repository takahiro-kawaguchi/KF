clear
close all

R0 = 0.1;
R1 = 0.2;
R2 = 1;
Cocv = 10;
C1 = 1;
C2 = 5;

Q = diag([1e-5, 2e-5, 2e-5]);
R = 1e-5;

N = 10000;
Ts = 0.1;

v = mvnrnd(zeros(1, 3), Q, N);
w = randn(N, 1) * sqrt(R);

A = eye(3) - diag([0, Ts/C1/R1, Ts/C2/R2]);
B = [Ts/Cocv; Ts/C1; Ts/C2];
C = [1 1 1];
D = R0;

% 実験データ生成
Y = zeros(N, 1);
U = randn(N, 1);
x = zeros(3, 1);
for k = 1:N
    u = U(k, :)';
    y = C*x + D*u + w(k, :)';
    x = A*x + B*u + v(k, :)';
    Y(k) = y;
end

evalfunc = @(parameters) calc_likelihood(parameters, U, Y, Ts);
options = optimset('PlotFcns',@optimplotfval);
x0 = [0.1, 0.1, 0.1, 1, 1, 1, 1e-2, 1e-2, 1e-2, 1e-3];
xtrue = [R0 R1 R2 Cocv C1 C2 diag(Q)', R];
xopt = fmincon(evalfunc, x0, [], [], [], [], zeros(size(x0)), [], [], options);

R0_est = xopt(1)
R1_est = xopt(2)
R2_est = xopt(3)
Cocv_est = xopt(4)
C1_est = xopt(5)
C2_est = xopt(6)

function out = calc_likelihood(parameters, U, Y, Ts)
R0 = parameters(1);
R1 = parameters(2);
R2 = parameters(3);
Cocv = parameters(4);
C1 = parameters(5);
C2 = parameters(6);
Q = diag(parameters(7:9));
R = parameters(10);

A = eye(3) - diag([0, Ts/C1/R1, Ts/C2/R2]);
B = [Ts/Cocv; Ts/C1; Ts/C2];
C = [1 1 1];
D = R0;

N = size(Y, 1);

xhat = zeros(3, 1);
P = 0.1*eye(3);
out = 0;
for k = 1:N
    y = Y(k);
    u = U(k);
    [xhat, P, K, likelihood] = kf_update_likelihood(xhat, P, y, u, C, D, R);
    [xhat, P] = kf_predict(xhat, P, u, A, B, eye(3), Q);
    if k > 100
    out = out + likelihood;
    end
end
end