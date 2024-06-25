clear
rng(123)
N = 1000;
a = 0.9;
b = 2;
sigma_w = 1;

y = 0;
A = eye(2);
B = eye(2);
Bu = zeros(2, 0);

C = [0 0];

xhat = zeros(2, 1);
P = eye(2)*1000;

% Q = diag([1e-2, 1e-3]);
Q = diag([1e-1, 1e-6]);
R = sigma_w;

Y = zeros(N, 1);
U = randn(N, 1);
X = zeros(N, 2);
Xhat = zeros(N, 2);


for k = 1:N
    X(k, :) = [a b];
    Y(k, :) = y;
    
    [xhat, P] = kf_update(xhat, P, y, C, R);
    Xhat(k, :) = xhat';

    [xhat, P] = kf_predict(xhat, P, [], A, Bu, B, Q);
    u = randn();
    U(k) = u;
    C = [y, u];
    y = a*y+b*u+randn()*sqrt(sigma_w);

    if k == N/2
       a = 0.5; 
    end
end

figure, plot([Xhat])
hold on
plot(X, 'k')