clear
rng(123)
% 初期設定
A = [1.5 -0.7; 1 0];
B = [0.5; -0.5];
Bu= [1; 0.5];
C = [1, 0];
Q = 10;
R = 1;
N = 1000;
P = eye(2);
xhat = zeros(2, 1);
x = zeros(2, 1);
u = randn(N, 1);
X = zeros(N, 2);
Xhat = zeros(N, 2);

% システムのシミュレーションと推定
for k=1:N
    y = C*x + sqrt(R)*randn();
    [xhat, P] = kf_update(xhat, P, y, C, R);
    Xhat(k, :) = xhat';
    X(k, :) = x';
    
    x = A*x + Bu*u(k) + B*sqrt(Q)*randn();
    [xhat, P] = kf_predict(xhat, P, u(k), A, Bu, B, Q);
end

figure, plot([Xhat(:, 1), X(:, 1)])
figure, plot([Xhat(:, 2), X(:, 2)])

