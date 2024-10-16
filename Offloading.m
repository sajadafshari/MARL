% Parameters
num_agents = 5; % Number of agents (LEO satellites)
episodes = 1000; % Number of training episodes
max_steps = 50; % Steps per episode
state_size = 11; % State space size (queue lengths from 0 to 10)
action_size = 3; % Actions: 1=Process locally, 2=Offload to neighbor, 3=Offload to ground
gamma = 0.99; % Discount factor
alpha = 0.1; % Learning rate
epsilon = 1.0; % Initial epsilon for epsilon-greedy policy
epsilon_min = 0.01;
epsilon_decay = 0.995;

% Environment Parameters
parameters.task_arrival_rate = 0.5; % Probability of new task arrival
parameters.max_queue_length = 10;
parameters.local_processing_time = 1;
parameters.local_energy_consumption = 2;
parameters.offload_processing_time = 1;
parameters.offload_energy_consumption = 1;
parameters.ground_processing_time = 1;
parameters.ground_energy_consumption = 0.5;
parameters.offload_delay = 2; % Time delay when offloading to neighbor
parameters.ground_delay = 3; % Time delay when offloading to ground
parameters.energy_weight = 0.1; % Weight of energy consumption in reward
parameters.communication_cost = -0.5; % Communication cost for CDRL

% Initialize Q-tables for MARL and CDRL
Q_marl = zeros(state_size, action_size, num_agents);
Q_cdrl = zeros(state_size, action_size);

% Train MARL and CDRL frameworks
rewards_marl = train_marl(Q_marl, parameters, episodes, max_steps, gamma, alpha, epsilon, epsilon_min, epsilon_decay, num_agents);
rewards_cdrl = train_cdrl(Q_cdrl, parameters, episodes, max_steps, gamma, alpha, epsilon, epsilon_min, epsilon_decay, num_agents);

% Plot the results
figure;
plot(1:episodes, rewards_marl, 'LineWidth', 1.5);
hold on;
plot(1:episodes, rewards_cdrl, 'LineWidth', 1.5);
xlabel('Episodes');
ylabel('Total Reward');
title('Comparison of MARL vs CDRL in LEO Satellite Offloading');
legend('MARL', 'CDRL');
grid on;

% LEO Environment function
function [next_state, reward] = leo_environment(state, action, agent, parameters)
    queue_length = state;
    % Simulate task arrival
    if rand < parameters.task_arrival_rate
        queue_length = min(queue_length + 1, parameters.max_queue_length);
    end
    % Process action
    if action == 1 % Process locally
        if queue_length > 0
            queue_length = queue_length - 1;
            processing_time = parameters.local_processing_time;
            energy_consumption = parameters.local_energy_consumption;
            delay = processing_time;
        else
            processing_time = 0;
            energy_consumption = 0;
            delay = 0;
        end
    elseif action == 2 % Offload to neighbor
        if queue_length > 0
            queue_length = queue_length - 1;
            processing_time = parameters.offload_processing_time;
            energy_consumption = parameters.offload_energy_consumption;
            delay = parameters.offload_delay;
        else
            processing_time = 0;
            energy_consumption = 0;
            delay = 0;
        end
    elseif action == 3 % Offload to ground
        if queue_length > 0
            queue_length = queue_length - 1;
            processing_time = parameters.ground_processing_time;
            energy_consumption = parameters.ground_energy_consumption;
            delay = parameters.ground_delay;
        else
            processing_time = 0;
            energy_consumption = 0;
            delay = 0;
        end
    else
        error('Invalid action');
    end
    % Compute reward (negative cost)
    reward = -(delay + parameters.energy_weight * energy_consumption);
    next_state = queue_length;
end

% MARL training function
function rewards = train_marl(Q_marl, parameters, episodes, max_steps, gamma, alpha, epsilon, epsilon_min, epsilon_decay, num_agents)
    rewards = zeros(1, episodes);
    for episode = 1:episodes
        total_reward = 0;
        % Initialize state for each agent
        states = zeros(1, num_agents); % Queue lengths, start at 0
        for step = 1:max_steps
            actions = zeros(1, num_agents);
            for agent = 1:num_agents
                state = states(agent) + 1; % Add 1 for 1-based indexing
                % Epsilon-greedy action selection
                if rand < epsilon
                    action = randi([1, 3]);
                else
                    [~, action] = max(Q_marl(state, :, agent));
                end
                actions(agent) = action;
            end
            % Simulate environment and update Q-tables
            rewards_step = zeros(1, num_agents);
            next_states = zeros(1, num_agents);
            for agent = 1:num_agents
                state = states(agent);
                action = actions(agent);
                [next_state, reward] = leo_environment(state, action, agent, parameters);
                rewards_step(agent) = reward;
                next_state_idx = next_state + 1;
                state_idx = state + 1;
                % Update Q-table
                Q_marl(state_idx, action, agent) = Q_marl(state_idx, action, agent) + alpha * ...
                    (reward + gamma * max(Q_marl(next_state_idx, :, agent)) - Q_marl(state_idx, action, agent));
                next_states(agent) = next_state;
            end
            total_reward = total_reward + sum(rewards_step);
            states = next_states;
        end
        rewards(episode) = total_reward;
        % Decay epsilon
        if epsilon > epsilon_min
            epsilon = epsilon * epsilon_decay;
        end
    end
end

% CDRL training function
function rewards = train_cdrl(Q_cdrl, parameters, episodes, max_steps, gamma, alpha, epsilon, epsilon_min, epsilon_decay, num_agents)
    rewards = zeros(1, episodes);
    for episode = 1:episodes
        total_reward = 0;
        % Initialize state for each agent
        states = zeros(1, num_agents); % Queue lengths, start at 0
        for step = 1:max_steps
            actions = zeros(1, num_agents);
            for agent = 1:num_agents
                state = states(agent) + 1; % Add 1 for 1-based indexing
                % Epsilon-greedy action selection
                if rand < epsilon
                    action = randi([1, 3]);
                else
                    [~, action] = max(Q_cdrl(state, :));
                end
                actions(agent) = action;
            end
            % Simulate environment and update Q-table
            rewards_step = zeros(1, num_agents);
            next_states = zeros(1, num_agents);
            for agent = 1:num_agents
                state = states(agent);
                action = actions(agent);
                [next_state, reward] = leo_environment(state, action, agent, parameters);
                % Add communication cost for centralized control
                reward = reward + parameters.communication_cost;
                rewards_step(agent) = reward;
                next_state_idx = next_state + 1;
                state_idx = state + 1;
                % Update centralized Q-table
                Q_cdrl(state_idx, action) = Q_cdrl(state_idx, action) + alpha * ...
                    (reward + gamma * max(Q_cdrl(next_state_idx, :)) - Q_cdrl(state_idx, action));
                next_states(agent) = next_state;
            end
            total_reward = total_reward + sum(rewards_step);
            states = next_states;
        end
        rewards(episode) = total_reward;
        % Decay epsilon
        if epsilon > epsilon_min
            epsilon = epsilon * epsilon_decay;
        end
    end
end
