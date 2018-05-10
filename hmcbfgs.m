function [ Samples, acprate, nfevals ] = hmcbfgs(f, N, n, sample_mem, L, Eps, gamma, debug)
% HMC Summary of this function goes here
% f: the target log density
% N: the number of samples
% n: the number of burn-in samples
% sample: k x dim matrix k is the number of chains in the ensemble
% L: the number of leaps
% Esp: the step size

d = size(sample_mem, 2);
m = size(sample_mem, 1) - 1;
Samples = {};
for i = 1 : m + 1
    Samples{i} = zeros(N / (m + 1), d);
end
nfevals = zeros(1, N);
acprate = 0;
B1 = eye(d) * gamma;

% Initalize memory excluding the first sample in the memory
sample = sample_mem(1, :);
x_mem = sample_mem(2 : m + 1, :);
g_mem = zeros(m, d);
E_mem = zeros(1, m);
for i = 1 : m
    [logp, g] = f(x_mem(i, :));
    E_mem(i) = -logp;
    g_mem(i, :) = -g;
end

for i = 1 : N + n
    if mod(i,500) == 0
        disp(['sampling iteration ' num2str(i)]);
    end
    
    % Precompute BFGS-conjugate matrices
    [P Q U T] = update_PQUT(x_mem, g_mem, E_mem, B1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     Refresh Momentum       %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    p = randn(1, d);
    p = Cz_product(U, T, B1, p');
    p = p';
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     Simulate Trajectory    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [X_path, E_path, K_path, nfevali g0 g] = leapfrog_BFGS(f, sample, p, ...
        L, Eps, P, Q, B1);
    if debug
    figure(1)
    subplot(131)
    plot(X_path(:,1), X_path(:,end), '.-');
    xlabel('x1')
    ylabel(['x' num2str(d)]);
    subplot(132)
    plot(X_path(:,1), X_path(:,2), '.-');
    xlabel('x1');
    ylabel('x2');
    subplot(133)
    plot(X_path(:,end-1), X_path(:,end), '.-');
    xlabel(['x' num2str(d-1)]);
    ylabel(['x' num2str(d)]);
    
    figure(2)
    hold off
    plot(E_path,'.-');
    hold on
    plot(K_path,'r.-');
    plot(E_path + K_path);
    grid on
    
    pause;
    end
    H = E_path(1) + K_path(1);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     Metroplis-Hasting      %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H_new = E_path(end) + K_path(end);
    
    dH = H_new - H;
    
    accept = 0;
    if dH < 0
        accept = 1;
    else
        if rand < exp( -dH )
            accept = 1;
        end
    end
    
    sample_next = x_mem(1, :);
    if accept
        sample = X_path(end, :);
        % update memory
        x_mem = [x_mem(2 : m, :); sample];
        g_mem = [g_mem(2 : m, :); g];
        E_mem = [E_mem(2 : m), E_path(end)];
    else
        % update memory
        x_mem = [x_mem(2 : m, :); sample];
        g_mem = [g_mem(2 : m, :); g0];
        E_mem = [E_mem(2 : m), E_path(1)];
    end
    sample = sample_next;
    
    if i > n
        ensemble_id = mod(i, m + 1);
        subiter_id = ceil((i - n) / (m + 1)); 
        Samples{ensemble_id + 1}(subiter_id, :) = sample;
        acprate = acprate + accept;
        nfevals(i - n) = nfevali;
    end
    
end

acprate = acprate / N;
end

function [ X_path E_path K_path nfevals g0 g] = leapfrog_BFGS(f, x, p,...
    L, Eps, P, Q, B1)
% The Leapfrog integerator for Hamiltonian Monte Carlo. It returns a
% fragment of the trajectory of Hamiltonian dynamics. The momentum should
% be drawn from the normal distribution.
%
% f: the log density of target distribution
% x: the initial location
% p: the initial momentum
% L: the number of function evaluation
% T: the maximum of simulation time
% *t0:
% *map:
% * denotes the output interface for phmc algorithm

L_rand = ceil(rand * L);
eps = (0.9 + rand / 5) * Eps;
X_path = zeros(L_rand + 1, length(x));
E_path = zeros(1, L_rand + 1);
K_path = zeros(1, L_rand + 1);
X_path(1, :) = x;
[E_path(1) grad_new] = f(x);
g0 = -grad_new;
E_path(1) = -E_path(1);
K_path(1) = 0.5 * p * Hz_product(P, Q, B1, p');

x_new = x;
p_new = p;
nfevals = 1;
for i = 2 : L_rand + 1
    p_new = p_new + 0.5 * eps * grad_new;
    x_new = x_new + eps * Hz_product(P, Q, B1, p_new')';
    [E_new, grad_new] = f(x_new);
    nfevals = nfevals + 1;
    p_new = p_new + 0.5 * eps * grad_new;
    
    X_path(i, :) = x_new;
    E_path(i) = -E_new;
    K_path(i) = 0.5 * p_new * Hz_product(P, Q, B1, p_new');
end

g = -grad_new;

end

function [P Q U T] = update_PQUT(x_mem, g_mem, E_mem, B1)
dim = size(x_mem, 2);
[~, o] = sort(E_mem, 'ascend');

S = [];
Y = [];
v_last = 1;
v_next = 2;
while v_next <= length(o)
    si = x_mem(v_next, :) - x_mem(v_last, :);
    yi = g_mem(v_next, :) - g_mem(v_last, :);
    if si * yi' > 0
        v_last = v_next;
        v_next = v_next + 1;
        S = [S, si'];
        Y = [Y, yi'];
    else
        v_next = v_next + 1;
    end
end

vn = size(S, 2);
P = zeros(dim, vn);
Q = zeros(dim, vn);
U = [];
T = [];
for i = 1 : vn
    syi = S(:, i)' * Y(:, i);
    Bsi = Bz_product(U, T, B1, S(:, i));
    sBsi = S(:, i)' * Bsi;
    P(:, i) = S(:, i) / syi;
    Q(:, i) = (syi / sBsi)^0.5 * Bsi + Y(:, i);
    T = [T, S(:, i) / sBsi];
    U = [U, (sBsi /  syi)^0.5 * Y(:, i) + Bsi];
end
end

function Cz = Cz_product(U, T, B1, z)
% BFGS-conjugate Hessian vector product
% z is the vector
% B = C C^T

C1 = B1.^0.5;
M = size(U, 2);
Ciz = C1 * z;
for i = 1 : M
    Ciz = Ciz - (T(:, i)' * Ciz) * U(:, i);
end
Cz = Ciz;
end

function CTz = CTz_product(U, T, B1, z)
% BFGS-conjugate Hessian vector product
% z is the vector
% B = C C^T

C1 = B1.^0.5;
M = size(U, 2);
Ciz = z;
for i = M : -1 : 1
    Ciz = Ciz - (U(:, i)' * Ciz) * T(:, i);
end
CTz = C1 * Ciz;
end

function Bz = Bz_product(U, T, B1, z)
% BFGS Hessian vector product
% z is the vector
% B = CC^T
if isempty(U)
    Bz = B1 * z;
    return;
end

% compute C^Tz
CTz = CTz_product(U, T, B1, z);
% compute Bz = CC^Tz
Bz = Cz_product(U, T, B1, CTz);
end

function Sz = Sz_product(P, Q, B1, z)
% BFGS-conjugate inverse Hessian vector product
% z is the vector
% H = SS^T

S1 = diag(diag(B1).^-0.5);
M = size(P, 2);
Siz = S1 * z;
for i = 1 : M
    Siz = Siz - (Q(:, i)' * Siz) * P(:, i);
end
Sz = Siz;
end

function STz = STz_product(P, Q, B1, z)
% BFGS-conjugate inverse Hessian vector product
% z is the vector
% H = SS^T

% compute S^Tz
S1 = diag(diag(B1).^-0.5);
M = size(P, 2);
Siz = z;
for i = M : -1 : 1
    Siz = Siz - (P(:, i)' * Siz) * Q(:, i);
end
STz = S1 * Siz;
end

function Hz = Hz_product(P, Q, B1, z)
% BFGS inverse Hessian vector product
% z is the vector
% H = SS^T

if isempty(P)
    Hz = inv(B1) * z;
end

% compute S^Tz
STz = STz_product(P, Q, B1, z);
% compute SS^Tz
Hz = Sz_product(P, Q, B1, STz);
end


