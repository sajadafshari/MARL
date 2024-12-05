clear; clc; close all;

% Parameters
num_satellites = 60;             % Number of satellites
num_gts = 20;                    % Number of ground terminals (GTs)
channels_per_satellite = 10;     % Available channels per satellite
power_per_mbps = 5;              % Power consumption per Mbps (Watts per Mbps)
base_power_budget = 500;         % Base power budget for each satellite
num_iterations = 5000;           % Number of iterations (episodes)

% Initialize variables
throughput_marl = zeros(1, num_iterations);
throughput_greedy = zeros(1, num_iterations);
power_marl = zeros(1, num_iterations);
power_greedy = zeros(1, num_iterations);
latency_marl = zeros(1, num_iterations);
latency_greedy = zeros(1, num_iterations);

% Initialize Q-tables for each satellite
% Q-tables: For each satellite, we have a Q-value for each (GT, Channel)
% Essentially, Q_tables{sat}(gt, ch)
q_tables = cell(1, num_satellites);
for sat = 1:num_satellites
    q_tables{sat} = zeros(num_gts, channels_per_satellite);  
end

% Q-learning (bandit) parameters
alpha = 0.01;        % Learning rate
epsilon_start = 1; % Initial exploration probability
epsilon_end = 0.001;  % Final exploration probability
epsilon_decay = (epsilon_end/epsilon_start)^(1/num_iterations);

% Generate consistent GT demands for fair comparison (stationary distribution)
% To see learning, let's make some GTs consistently have higher demands.
% For example, half of the GTs have higher mean demand than the others.
high_demand_gts = 1:(num_gts/2);
low_demand_gts = (num_gts/2+1):num_gts;

% Mean demands
mean_high = 15;  % Higher demand mean
mean_low = 7;    % Lower demand mean

% For each iteration, we sample demands from a distribution
% This keeps some structure so that learning can occur:
% - The first half of GTs consistently offer higher demands on average.
% - The second half offer lower demands on average.
gt_demand_all = zeros(num_iterations, num_gts);
for i = 1:num_iterations
    gt_demand_all(i, high_demand_gts) = randi([mean_high-2, mean_high+2], 1, length(high_demand_gts));
    gt_demand_all(i, low_demand_gts) = randi([mean_low-2, mean_low+2], 1, length(low_demand_gts));
end

% Main simulation loop
epsilon = epsilon_start;
for iter = 1:num_iterations
    % Dynamically vary power budgets slightly
    power_budget = base_power_budget + randi([50, 100], 1, num_satellites);

    %% Greedy Algorithm
    total_throughput_greedy = 0;
    total_power_greedy = 0;
    gt_demand_iter = gt_demand_all(iter, :);
    for sat = 1:num_satellites
        gt_demand = gt_demand_iter;  
        total_power_sat = 0;
        total_throughput_sat = 0;
        power_limit = power_budget(sat);

        % Sort GTs based on demand (highest first)
        [sorted_demand, gt_indices] = sort(gt_demand, 'descend');
        ch = 1;
        idx = 1;
        while ch <= channels_per_satellite && idx <= num_gts
            demand = sorted_demand(idx);
            power_consumed = demand * power_per_mbps;
            if total_power_sat + power_consumed <= power_limit
                total_power_sat = total_power_sat + power_consumed;
                total_throughput_sat = total_throughput_sat + demand;
            end
            ch = ch + 1;
            idx = idx + 1;
        end
        total_throughput_greedy = total_throughput_greedy + total_throughput_sat;
        total_power_greedy = total_power_greedy + total_power_sat;
    end
    throughput_greedy(iter) = total_throughput_greedy;
    power_greedy(iter) = total_power_greedy;
    latency_greedy(iter) = num_gts / (total_throughput_greedy + 1e-5);  

    %% MARL Simulation (Bandit Approach)
    total_throughput_marl = 0;
    total_power_marl = 0;
    gt_demand_iter_marl = gt_demand_all(iter, :);
    
    for sat = 1:num_satellites
        gt_demand = gt_demand_iter_marl;
        total_power_sat = 0;
        total_throughput_sat = 0;
        power_limit = power_budget(sat);
        q_table = q_tables{sat};  % Retrieve the Q-table

        for ch = 1:channels_per_satellite
            % Epsilon-greedy action selection based on current Q-values
            if rand < epsilon
                action = randi(num_gts);  % Exploration
            else
                [~, action] = max(q_table(:, ch));  % Exploitation
            end

            demand = gt_demand(action);
            power_consumed = demand * power_per_mbps;

            if (demand > 0) && (total_power_sat + power_consumed <= power_limit)
                % Valid action: serve this GT
                total_power_sat = total_power_sat + power_consumed;
                total_throughput_sat = total_throughput_sat + demand;
                gt_demand(action) = 0;  % This GT served for this iteration
                reward = demand;        % Reward = served demand
            else
                % Invalid action (no throughput gained)
                reward = 0;
            end

            % Update Q-value (bandit update: no future states)
            current_q = q_table(action, ch);
            q_table(action, ch) = current_q + alpha * (reward - current_q);
        end

        % Save the updated Q-table
        q_tables{sat} = q_table;

        total_throughput_marl = total_throughput_marl + total_throughput_sat;
        total_power_marl = total_power_marl + total_power_sat;
    end

    throughput_marl(iter) = total_throughput_marl;
    power_marl(iter) = total_power_marl;
    latency_marl(iter) = num_gts / (total_throughput_marl + 1e-5); 

    % Decay epsilon over time to allow more exploitation as we learn
    epsilon = epsilon * epsilon_decay;
end

%% Plot results
figure;
plot(1:num_iterations, throughput_marl, 'r', 'LineWidth', 2, 'DisplayName', 'MARL Throughput');
hold on;
plot(1:num_iterations, throughput_greedy, 'b', 'LineWidth', 2, 'DisplayName', 'Greedy Throughput');
xlabel('Iterations', 'FontSize', 12);
ylabel('Total Throughput (Mbps)', 'FontSize', 12);
legend('show', 'FontSize', 12);
title('Throughput Comparison: MARL vs Greedy', 'FontSize', 12);
set(gca, 'FontSize', 12);
grid on;

figure;
plot(1:num_iterations, power_marl, 'r', 'LineWidth', 2, 'DisplayName', 'MARL Power Usage');
hold on;
plot(1:num_iterations, power_greedy, 'b', 'LineWidth', 2, 'DisplayName', 'Greedy Power Usage');
xlabel('Iterations', 'FontSize', 12);
ylabel('Total Power Usage (Watts)', 'FontSize', 12);
legend('show', 'FontSize', 12);
title('Power Usage Comparison: MARL vs Greedy', 'FontSize', 12);
set(gca, 'FontSize', 12);
grid on;

figure;
plot(1:num_iterations, latency_marl, 'r', 'LineWidth', 2, 'DisplayName', 'MARL Latency');
hold on;
plot(1:num_iterations, latency_greedy, 'b', 'LineWidth', 2, 'DisplayName', 'Greedy Latency');
xlabel('Iterations', 'FontSize', 12);
ylabel('Latency (GTs per Mbps)', 'FontSize', 12);
legend('show', 'FontSize', 12);
title('Latency Comparison: MARL vs Greedy', 'FontSize', 12);
set(gca, 'FontSize', 12);
grid on;
