% Parameters
num_satellites = 60;                 % Number of satellites
num_gts = 20;                        % Number of ground terminals (GTs)
channels_per_satellite = 10;         % Available channels per satellite
power_per_mbps = 5;                  % Power consumption per Mbps (Watts per Mbps)
power_budget = 500 + randi([50, 100], 1, num_satellites);   % Dynamic power budget for each satellite in Watts
num_iterations = 5000;               % Number of iterations

% Initialize variables
throughput_marl = zeros(1, num_iterations);
throughput_greedy = zeros(1, num_iterations);
power_marl = zeros(1, num_iterations);
power_greedy = zeros(1, num_iterations);
latency_marl = zeros(1, num_iterations);
latency_greedy = zeros(1, num_iterations);

% Initialize Q-tables for each satellite
q_tables = cell(1, num_satellites);
for sat = 1:num_satellites
    q_tables{sat} = zeros(num_gts, channels_per_satellite);  % Q-table for each satellite
end

% Q-learning parameters
alpha = 0.1;      % Learning rate
gamma = 0.9;      % Discount factor
epsilon = 0.1;    % Exploration probability

% Generate consistent GT demand for fair comparison
gt_demand_all = randi([5 15], [num_iterations, num_gts]);  % Demand between 5 and 15 Mbps

% Simulation loop
for iter = 1:num_iterations
    % --- Greedy Algorithm Simulation ---
    total_throughput_greedy = 0;
    total_power_greedy = 0;
    for sat = 1:num_satellites
        gt_demand = gt_demand_all(iter, :);  % Consistent demand
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
    latency_greedy(iter) = num_gts / (total_throughput_greedy + 1e-5);  % Latency as GTs per throughput

    % --- MARL Simulation ---
    total_throughput_marl = 0;
    total_power_marl = 0;
    for sat = 1:num_satellites
        gt_demand = gt_demand_all(iter, :);  % Consistent demand
        total_power_sat = 0;
        total_throughput_sat = 0;
        power_limit = power_budget(sat);
        q_table = q_tables{sat};  % Retrieve the Q-table

        for ch = 1:channels_per_satellite
            % Epsilon-greedy action selection
            if rand < epsilon
                action = randi(num_gts);  % Exploration
            else
                [~, action] = max(q_table(:, ch));  % Exploitation
            end

            % Apply action if GT demand is not zero and power allows
            demand = gt_demand(action);
            power_consumed = demand * power_per_mbps;
            if demand > 0 && total_power_sat + power_consumed <= power_limit
                total_power_sat = total_power_sat + power_consumed;
                total_throughput_sat = total_throughput_sat + demand;
                gt_demand(action) = 0;  % GT has been served
                reward = demand;  % Reward is the served demand
            else
                reward = -1;  % Penalty for invalid action
            end

            % Update Q-table
            max_future_q = max(q_table(:, ch));
            current_q = q_table(action, ch);
            q_table(action, ch) = current_q + alpha * (reward + gamma * max_future_q - current_q);
        end

        % Save the updated Q-table
        q_tables{sat} = q_table;

        total_throughput_marl = total_throughput_marl + total_throughput_sat;
        total_power_marl = total_power_marl + total_power_sat;
    end
    throughput_marl(iter) = total_throughput_marl;
    power_marl(iter) = total_power_marl;
    latency_marl(iter) = num_gts / (total_throughput_marl + 1e-5);  % Latency as GTs per throughput
end

% Plot comparison of throughput with font size 16 and units in Mbps
figure;
plot(1:num_iterations, throughput_marl, 'r', 'LineWidth', 2, 'DisplayName', 'MARL Throughput');
hold on;
plot(1:num_iterations, throughput_greedy, 'b', 'LineWidth', 2, 'DisplayName', 'Greedy Throughput');
xlabel('Iterations', 'FontSize', 12);
ylabel('Total Throughput (Mbps)', 'FontSize', 12); % Throughput is already in Mbps
legend('show', 'FontSize', 12);
title('Throughput Comparison: MARL vs Greedy', 'FontSize', 12);
set(gca, 'FontSize', 12);
grid on;
saveas(gcf, 'throughput_comparison_with_units.png');

% Plot comparison of power usage with font size 16
figure;
plot(1:num_iterations, power_marl, 'r', 'LineWidth', 2, 'DisplayName', 'MARL Power Usage');
hold on;
plot(1:num_iterations, power_greedy, 'b', 'LineWidth', 2, 'DisplayName', 'Greedy Power Usage');
xlabel('Iterations', 'FontSize', 12);
ylabel('Total Power Usage (Watts)', 'FontSize', 12); % Power usage in Watts
legend('show', 'FontSize', 12);
title('Power Usage Comparison: MARL vs Greedy', 'FontSize', 12);
set(gca, 'FontSize', 12);
grid on;
saveas(gcf, 'power_usage_comparison_with_units.png');

% Plot comparison of latency with font size 16 (optional if needed for completeness)
figure;
plot(1:num_iterations, latency_marl, 'r', 'LineWidth', 2, 'DisplayName', 'MARL Latency');
hold on;
plot(1:num_iterations, latency_greedy, 'b', 'LineWidth', 2, 'DisplayName', 'Greedy Latency');
xlabel('Iterations', 'FontSize', 12);
ylabel('Latency (GTs per Mbps)', 'FontSize', 12); % Latency units adjusted
legend('show', 'FontSize', 12);
title('Latency Comparison: MARL vs Greedy', 'FontSize', 12);
set(gca, 'FontSize', 12);
grid on;
saveas(gcf, 'latency_comparison_with_units.png');
