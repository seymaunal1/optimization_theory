% Function

f = @(x) (x(1) + 10*x(2))^2 + 5*((x(3) - x(4))^2) + (x(2) - 2*x(3))^4 + 10*((x(1)-x(4))^4);

x_initial = [3; -1; 0; 1];

terminal_criteria = 10^(-2);

syms x1 x2 x3 x4
f_sym = (x1 + 10*x2)^2 + 5*(x3 - x4)^2 + (x2 - 2*x3)^4 + 10*(x1 - x4)^4;
vars = [x1; x2; x3; x4];

grad_f = gradient(f_sym, vars);   % Gradient vector
H_f = hessian(f_sym, vars);       % Hessian matrix

Gradient = matlabFunction(grad_f, 'Vars', {vars});
Hessian = matlabFunction(H_f, 'Vars', {vars});

%% P1

max_iters = 200000;
alphas = [10^(-3) 10^(-4)];

grad_norms_all = cell(2, 1);
X_list = cell(2, 1);
fx_list = cell(2, 1);
stop_iter = NaN;

figure; hold on;
colors = lines(length(alphas));
legend_labels = {};

for i = 1:length(alphas)
    alpha = alphas(i);
    x = x_initial;
    grad_norms = [];
    X = x;
    fx = f(x);
    for k = 1:max_iters
        g = Gradient(x);
    
        norm_grad = norm(g);
        grad_norms(end+1) = norm_grad;

        if norm_grad <= terminal_criteria
            stop_iter = k;
            fprintf('P1.%d Termination criteria met at iteration %d\n', i, k);
            fprintf('P1.%d Eventual function value %d\n', i, f(x));
            break
        end
    
        d = -g;
        x = x + alpha * d;
        X(:,end+1) = x;
        fx(end+1) = f(x);
    end

    X_list{i} = X;
    fx_list{i} = fx;

    grad_norms_all{i} = grad_norms;
    plot(0:length(grad_norms)-1, grad_norms, 'LineWidth', 2, 'Color', colors(i,:));
    legend_labels{end+1} = sprintf('\\alpha = %.0e', alpha);
end

xlabel('Iteration');
ylabel('||\nabla f(x)||');
title('P1: Gradient Norm vs Iteration (Log Scale)');
legend(legend_labels);
grid on;
set(gca, 'YScale', 'log');

hold off;


for i = 1:2
   figure;
   X = X_list{i};
   fx = fx_list{i};
   for j = 1:4
       subplot(5,1,j)
       plot(0:length(X)-1, X(j, :), 'LineWidth', 2, 'Color', colors(i,:));
       ylabel(['x_', num2str(j)]);
   end
   subplot(5,1,5)
   plot(0:length(fx)-1, fx, 'LineWidth', 2, 'Color', colors(i,:));
   ylabel('f(x)');
   xlabel('Iteration');
   sgtitle(['x_1, x_2, x_3, x_4, f(x) vs Iteration (\alpha=',num2str(alphas(i)),')']);
end


%% P2

x = x_initial;
X = x;
F = f(x);
alphas = [];
grad_norms = [];
stop_iter = NaN;
line_search_iters = [];

% Golden section function
function [alpha_opt, iter_count] = golden_section(phi, a, b)
    tau = (sqrt(5) - 1) / 2;
    eps = 1e-5;

    c = b - tau * (b - a);
    d = a + tau * (b - a);
    iter_count = 0;

    while (b - a) > eps
        iter_count = iter_count + 1;
        if phi(c) < phi(d)
            b = d;
        else
            a = c;
        end
        c = b - tau * (b - a);
        d = a + tau * (b - a);
    end

    alpha_opt = (a + b) / 2;
end

for k = 1:1000
    g = Gradient(x);  


    grad_norms(end+1) = norm(g);

    if norm(g) < terminal_criteria
        stop_iter = k;
        fprintf('P2 Termination criteria met at iteration %d\n', k);
        fprintf('P2 Eventual function value %d\n', f(x));
        break;
    end

    d = -g;
    phi = @(a) f(x + a * d);
    [alpha, line_search_iter] = golden_section(phi, 0, 1e-1);                 
    line_search_iters(end+1) = line_search_iter;
    
    x = x + alpha * d;

    X(:, end+1) = x;
    F(end+1) = f(x);
end

figure;
plot(1:length(grad_norms), grad_norms, 'LineWidth', 1);
xlabel('Iteration');
ylabel('||\nabla f(x)||');
title('P2: Gradient Norm vs Iteration (Log Scale)');
grid on;
set(gca, 'YScale', 'log');


figure; 
for i = 1:4
    subplot(5,1,i);
    plot(0:length(X)-1, X(i,:), '-', 'LineWidth', 1); 
    grid on;
    ylabel(['x_', num2str(i)]);
end

subplot(5,1,5);
plot(0:length(X)-1, F, '-', 'LineWidth', 1);
grid on;
ylabel('f(x)', 'FontSize', 10);
xlabel('Iteration k', 'FontSize', 10);
set(gca, 'FontSize', 10);

sgtitle('x_1, x_2, x_3, x_4, f(x) vs Iteration');


fprintf('P2 Sum of Line Search Iterations %d\n', sum(line_search_iters));
fprintf('P2 Last Line Search Iterations %d\n', line_search_iters(end));



%% P3

max_iters = 1000;
x = x_initial;

epsilon = 0.2;
eta = 0.8;
tau_1 = 0.5;
tau_2 = 1.5;
alpha_init = 1;

grad_norms_p3 = [];
X_p3 = x;
fx_p3 = f(x);
stop_iter = NaN;
line_search_iters = [];

function [alpha, iter_count] = armijo_goldstein(phi, alpha_init, f_curr, dot_product, epsilon, eta, tau_1, tau_2)
    alpha = alpha_init;
    iter_count = 0;
    while phi(alpha) > f_curr + epsilon * alpha * dot_product
        alpha = tau_1 * alpha;
        iter_count = iter_count +1;
    end

    while phi(alpha) < f_curr + eta * alpha * dot_product
        alpha = tau_2 * alpha;
        iter_count = iter_count + 1;
    end
end

for k = 1:max_iters
    g = Gradient(x);
    norm_grad = norm(g);
    grad_norms_p3(end+1) = norm_grad;

    if norm_grad <= terminal_criteria
        stop_iter = k;
        fprintf('P3 Termination criteria met at iteration %d\n', k);
        fprintf('P3 Eventual function value %d\n', f(x));
        break;
    end

    d = -g;
    f_curr = f(x);
    dot_prod = g' * d;
    phi = @(a) f(x + a * d);
    
    [alpha, line_search_iter] = armijo_goldstein(phi, alpha_init, f_curr, dot_prod, epsilon, eta, tau_1, tau_2);
    line_search_iters(end+1) = line_search_iter;

    x = x + alpha * d;
    X_p3(:, end+1) = x;
    fx_p3(end+1) = f(x);
end

figure;
plot(0:length(grad_norms_p3)-1, grad_norms_p3, 'LineWidth', 2);
xlabel('Iteration');
ylabel('||\nabla f(x)||');
title('P3: Gradient Norm vs Iteration (Log Scale)');
grid on;
set(gca, 'YScale', 'log');

figure;
for j = 1:4
    subplot(5,1,j)
    plot(0:size(X_p3, 2)-1, X_p3(j, :), 'LineWidth', 2);
    ylabel(['x_', num2str(j)]);
end
subplot(5,1,5)
plot(0:length(fx_p3)-1, fx_p3, 'LineWidth', 2);
ylabel('f(x)');
xlabel('Iteration');
sgtitle('x_1, x_2, x_3, x_4, f(x) vs Iteration')

fprintf('P3 Sum of Line Search Iterations %d\n', sum(line_search_iters));
fprintf('P3 Last Line Search Iterations %d\n', line_search_iters(end));

%% P4

x = x_initial;
max_iters = 100;

X_p4 = x;
fx_p4 = f(x);
grad_norms_p4 = [];
stop_iter = NaN;

for k = 1:max_iters
    g = Gradient(x);
    H = Hessian(x);
    norm_grad = norm(g);
    grad_norms_p4(end+1) = norm_grad;

    if norm_grad <= terminal_criteria
        stop_iter = k;
        fprintf('P4 Termination criteria met at iteration %d\n', k);
        fprintf('P4 Eventual function value %d\n', f(x));
        break;
    end

    d = -H \ g;
    x = x + d;

    X_p4(:, end+1) = x;
    fx_p4(end+1) = f(x);
end

figure;
semilogy(0:length(grad_norms_p4)-1, grad_norms_p4, 'LineWidth', 2);
xlabel('Iteration');
ylabel('||\nabla f(x)||');
title('P4: Gradient Norm vs Iteration (Log Scale)');
grid on;


figure;
for j = 1:4
    subplot(5,1,j)
    plot(0:size(X_p4, 2)-1, X_p4(j, :), 'LineWidth', 2);
    ylabel(['x_', num2str(j)]);
end
subplot(5,1,5)
plot(0:length(fx_p4)-1, fx_p4, 'LineWidth', 2);
ylabel('f(x)');
xlabel('Iteration');
sgtitle('x_1, x_2, x_3, x_4, f(x) vs Iteration')


%% P5

x = x_initial;
X = x;
F = f(x);
line_search_iters = [];
grad_norms = [];
stop_iter = NaN;

for k = 1:1000
    g = Gradient(x);
    H = Hessian(x);
    grad_norms(end+1) = norm(g);

    if norm(g) < terminal_criteria 
        fprintf('P5 Termination criteria met at iteration %d\n', k);
        fprintf('P5 Eventual function value %d\n', f(x));
        stop_iter = k;
        break;
    end

    d = -H \ g;
    
    phi = @(a) f(x + a * d);
    [alpha, line_search_iter] = golden_section(phi,0,1);
    line_search_iters(end+1) = line_search_iter;

    x = x + alpha * d;
    X(:, end+1) = x;
    F(end+1) = f(x);
end


figure;
plot(1:length(grad_norms), grad_norms, 'LineWidth', 1.2);
xlabel('Iteration'); 
ylabel('||\nabla f(x)||');
title('P5: Gradient Norm vs Iteration (Log Scale)'); 
grid on;
set(gca,'Yscale', 'log')

figure;
for i = 1:4
    subplot(5,1,i);
    plot(0:length(X)-1, X(i,:), 'LineWidth', 2);
    ylabel(sprintf('x_%d', i)); grid on;
end
subplot(5,1,5);
plot(0:length(X)-1, F, 'LineWidth', 1.2);
ylabel('f(x)'); 
xlabel('Iteration'); 
grid on;
sgtitle('x_1, x_2, x_3, x_4, f(x) vs Iteration');

fprintf('P5 Sum of Line Search Iterations %d\n', sum(line_search_iters));
fprintf(['P5 Last Line Search Iterations %d\n'], line_search_iters(end));

%% P6

x = x_initial; 
X = x;
F = f(x);
grad_norms = [];
line_search_iters = [];
stop_iter = NaN;

max_iter = 1000;
epsilon = 0.4; 
eta = 1 - epsilon;
tau_1 = 0.5;
tau_2 = 1.5;
alpha_init = 1;

for k = 1:max_iter
    g = Gradient(x);
    H = Hessian(x);

    grad_norms(end+1) = norm(g);
    
    if norm(g) < terminal_criteria
        fprintf("P6 Termination criteria met at iteration %d\n", k);
        fprintf('P6 Eventual function value %d\n', f(x));
        stop_iter = k;
        break;
    end

    d = -H \ g;
    f_curr = f(x);
    dot_prod = g' * d;
    phi = @(a) f(x + a * d);

    [alpha, line_search_iter] = armijo_goldstein(phi, alpha_init, f_curr, dot_prod, epsilon, eta, tau_1, tau_2);
    line_search_iters(end+1) = line_search_iter;
    x = x + alpha * d;

    X(:, end+1) = x;
    F(end+1) = f(x);
end


figure;
plot(1:length(grad_norms), grad_norms, '-o', 'LineWidth', 1.2);
xlabel('Iteration');
ylabel('||\nabla f(x)||');
title('P6: Gradient Norm vs Iteration (Log Scale)');
set(gca, 'Yscale', 'log')
grid on;

figure;
for i = 1:4
    subplot(5,1,i);
    plot(0:length(X)-1, X(i,:), '-', 'LineWidth', 1.2);
    ylabel(sprintf('x_%d', i));
    grid on;
end
subplot(5,1,5);
plot(0:length(X)-1, F, '-', 'LineWidth', 1.2);
ylabel('f(x)');
xlabel('Iteration');
grid on;
sgtitle('x_1, x_2, x_3, x_4, f(x) vs Iteration');

fprintf('P6 Sum of Line Search Iterations %d\n', sum(line_search_iters));
fprintf('P6 Last Line Search Iterations %d\n', line_search_iters(end));

