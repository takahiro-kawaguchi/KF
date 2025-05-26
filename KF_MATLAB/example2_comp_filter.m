clear
rng(123)
% 初期設定
A = [1 0; 0 1];
B = eye(2);
Q = diag([5, 1]);
C = [1 0; 1 1];
sigma_w = 10^2;
sigma_w2 = 0.0;
R = diag([sigma_w, sigma_w2]);

xhat = zeros(2, 1);
P = eye(2)*10;

ehat = 0;
Pe = 10;

z = 10;
e = 5;

N = 1000;
Z = zeros(N, 1);

u = randn(N, 1);
X = zeros(N, 2);
Y = zeros(N, 2);
Xhat = zeros(N, 2);

Ehat = zeros(N, 1);

% システムのシミュレーションと推定
for k=1:N
    
    Z(k) = z;
    y1 = z + sqrt(sigma_w)*randn();
    y2 = z + e + sqrt(sigma_w2)*randn();
    y = [y1; y2];
    Y(k, :) = y';

    [xhat, P, K] = kf_update(xhat, P, y, C, R);
    Xhat(k, :) = xhat';
    
    [ehat, Pe] = kf_update(ehat, Pe, (y1-y2), -1, sigma_w);
    Ehat(k, :) = ehat';

    X(k, :) = [z, e];
    
    
    z = z+2*cos(0.05*k)+sqrt(5)*randn();
    e = e + randn();
    
    [xhat, P] = kf_predict(xhat, P, [], A, [], B, Q);
    [ehat, Pe] = kf_predict(ehat, Pe, [], 1, [], 1, Q(2, 2));
end

figure, plot([Y(:, 1), X(:, 1)])
figure, plot([Y(:, 2), X(:, 1)])
figure, plot([Xhat(:, 1), X(:, 1)])
% figure, plot([Xhat(:, 2), X(:, 2)])

f = figure, plot([Xhat(:, 1), Y(:, 2)-Ehat, X(:, 1)])

p = tools.plot_pptx()
p.add_plot(f, 1, 1, 1, 2)

figure, plot([Xhat(:, 1)-X(:,1), Y(:, 2)-Ehat-X(:, 1)])