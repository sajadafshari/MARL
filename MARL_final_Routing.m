% MATLAB Code for Enhanced MARL-Based Routing in LEO Satellite Networks
clear; clc; close all;

%% Check MATLAB Version Compatibility
if verLessThan('matlab', '9.7')
    error('This code requires MATLAB R2019b or later.');
end

%% Simulation Parameters
N = 60; % Number of satellites
E = 5000; % Number of episodes
alpha = 0.01; % Learning rate
gamma = 0.95; % Discount factor
epsilon = 1.0; % Initial exploration rate
epsilon_min = 0.01; % Minimum exploration rate
epsilon_decay = 0.995; % Decay rate of epsilon
positiveReward = 100; % Positive reward for reaching the destination
stepPenalty = 0; % No penalty for steps
loopPenalty = -50; % Penalty for loops
maxHops = N; % Maximum number of hops
destination = N; % Destination satellite
T_update = 100; % Network conditions update interval

%% Initialize satellite network
[satellitePositions, neighbors] = initializeSatellites(N);
linkDistances = calculateLinkDistances(satellitePositions, neighbors);
maxDistance = max(cellfun(@max, linkDistances)); % For normalization

%% Initialize Q-network
stateSize = 2 + max(cellfun(@length, neighbors)); % [distanceToDestination, queueUtilization, linkDelays]
actionSize = max(cellfun(@length, neighbors));
[QNetwork, dlnet, dlnetTarget] = initializeQNetwork(stateSize, actionSize);

%% Initialize experience replay buffer
replayBuffer = {};
bufferSize = 10000;
batchSize = 128; % Batch size

%% Simulation results storage
totalDelays_MARL = zeros(E,1);
totalDelays_SP = zeros(E,1);
totalHops_MARL = zeros(E,1);
totalHops_SP = zeros(E,1);

%% Main Simulation Loop
for episode = 1:E
    % Update network conditions periodically
    if mod(episode, T_update) == 1
        [linkDelays, queueUtilizations] = updateNetworkConditions(N, neighbors, linkDistances);
        maxDelay = max(cellfun(@max, linkDelays)); % For normalization
    end
    epsilon = max(epsilon_min, epsilon * epsilon_decay);

    % MARL-Based Routing
    currentNode = 1; % Source satellite
    hops_MARL = 0;
    delay_MARL = 0;
    visitedNodes = [currentNode];

    for t = 1:maxHops
        % Calculate distance to destination
        distanceToDestination = norm(satellitePositions(currentNode,:) - satellitePositions(destination,:));

        % Observe state
        state = constructState(currentNode, distanceToDestination, queueUtilizations, linkDelays, neighbors, stateSize, maxDistance, maxDelay);

        % Select action using epsilon-greedy policy
        if rand < epsilon
            action = randi(length(neighbors{currentNode}));
        else
            q_values = predictQValues(dlnet, state);
            % Mask invalid actions
            q_values = q_values(1:length(neighbors{currentNode}));
            [~, actionIdx] = max(q_values);
            action = actionIdx;
        end
        nextNode = neighbors{currentNode}(action);

        % Receive reward
        delay = linkDelays{currentNode}(action);

        if nextNode == destination
            reward = positiveReward - delay;
            done = true;
        elseif ismember(nextNode, visitedNodes)
            reward = loopPenalty - delay;
            done = true;
        else
            % Reward is negative delay only
            reward = -delay;
            done = false;
        end

        % Observe next state
        nextState = constructState(nextNode, distanceToDestination, queueUtilizations, linkDelays, neighbors, stateSize, maxDistance, maxDelay);

        % Store experience
        experience = {state, action, reward, nextState, done};
        replayBuffer = storeExperience(replayBuffer, experience, bufferSize);

        % Train Q-network
        if length(replayBuffer) >= batchSize
            minibatch = datasample(replayBuffer, batchSize, 'Replace', false);
            dlnet = trainQNetwork(dlnet, dlnetTarget, minibatch, gamma, alpha);
        end

        % Update target network periodically
        if mod(episode, 50) == 0
            dlnetTarget = dlnet;
        end

        % Accumulate delay and hops
        delay_MARL = delay_MARL + delay;
        hops_MARL = hops_MARL + 1;

        if done
            break;
        else
            % Move to next node
            currentNode = nextNode;
            visitedNodes = [visitedNodes, currentNode];
        end
    end

    totalDelays_MARL(episode) = delay_MARL;
    totalHops_MARL(episode) = hops_MARL;

    % Shortest-Path Routing
    [path_SP, delay_SP] = dijkstraRouting(1, destination, linkDelays, neighbors);
    totalDelays_SP(episode) = delay_SP;
    totalHops_SP(episode) = length(path_SP) - 1;

    % Display progress every 500 episodes
    if mod(episode, 500) == 0
        fprintf('Episode %d/%d completed.\n', episode, E);
    end
end

%% Plot results
windowSize = 100; % Moving average window size

figure;
plot(1:E, movmean(totalDelays_MARL, windowSize), 'r', 'LineWidth', 2);
hold on;
plot(1:E, movmean(totalDelays_SP, windowSize), 'b', 'LineWidth', 2);
xlabel('Episode');
ylabel('Total Communication Delay');
legend('MARL-Based Routing', 'Shortest-Path Routing');
title('Comparison of Total Communication Delay');

figure;
plot(1:E, movmean(totalHops_MARL, windowSize), 'r', 'LineWidth', 2);
hold on;
plot(1:E, movmean(totalHops_SP, windowSize), 'b', 'LineWidth', 2);
xlabel('Episode');
ylabel('Number of Hops');
legend('MARL-Based Routing', 'Shortest-Path Routing');
title('Comparison of Number of Hops');

%% Function Definitions

function [satellitePositions, neighbors] = initializeSatellites(N)
    % Initialize satellite positions in a grid for simplicity
    gridSize = ceil(sqrt(N));
    [X, Y] = meshgrid(1:gridSize, 1:gridSize);
    satellitePositions = [X(:), Y(:), zeros(gridSize^2,1)];
    satellitePositions = satellitePositions(1:N,:);
    neighbors = cell(N,1);
    for i = 1:N
        neighbors{i} = findNeighborsGrid(i, satellitePositions, N);
    end
end

function neighborList = findNeighborsGrid(i, positions, N)
    % Find neighbors in a grid (up, down, left, right)
    neighborList = [];
    position = positions(i,1:2);
    for j = 1:N
        if i ~= j
            neighborPosition = positions(j,1:2);
            if norm(position - neighborPosition) == 1
                neighborList(end+1) = j;
            end
        end
    end
end

function linkDistances = calculateLinkDistances(positions, neighbors)
    % Calculate distances between neighboring satellites
    N = size(positions,1);
    linkDistances = cell(N,1);
    for i = 1:N
        linkDistances{i} = zeros(length(neighbors{i}),1);
        for j = 1:length(neighbors{i})
            neighborIdx = neighbors{i}(j);
            distance = norm(positions(i,:) - positions(neighborIdx,:));
            linkDistances{i}(j) = distance;
        end
    end
end

function [QNetwork, dlnet, dlnetTarget] = initializeQNetwork(stateSize, actionSize)
    % Initialize Q-network using dlnetwork
    layers = [
        featureInputLayer(stateSize)
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(actionSize)];
    lgraph = layerGraph(layers);
    dlnet = dlnetwork(lgraph);
    dlnetTarget = dlnet;
    QNetwork = []; % Placeholder for compatibility
end

function state = constructState(node, distanceToDestination, queueUtilizations, linkDelays, neighbors, stateSize, maxDistance, maxDelay)
    % Construct state vector with normalization
    state = zeros(stateSize,1);
    state(1) = distanceToDestination / maxDistance; % Normalized distance
    state(2) = queueUtilizations(node); % Already between 0 and 1
    numNeighbors = length(neighbors{node});
    delays = linkDelays{node} / maxDelay; % Normalized delays
    state(3:numNeighbors+2) = delays;
end

function replayBuffer = storeExperience(replayBuffer, experience, bufferSize)
    % Store experience in replay buffer
    if length(replayBuffer) >= bufferSize
        replayBuffer(1) = []; % Remove oldest experience
    end
    replayBuffer{end+1} = experience; % Append new experience
end

function dlnet = trainQNetwork(dlnet, dlnetTarget, minibatch, gamma, alpha)
    % Train Q-network using experiences from minibatch
    numExperiences = length(minibatch);
    stateSize = size(minibatch{1}{1},1);
    actionSize = dlnet.Layers(end).OutputSize;
    inputs = zeros(stateSize, numExperiences);
    targets = zeros(actionSize, numExperiences);
    for i = 1:numExperiences
        experience = minibatch{i};
        state = experience{1};
        action = experience{2};
        reward = experience{3};
        nextState = experience{4};
        done = experience{5};
        q_values = predictQValues(dlnet, state);
        if done
            q_values(action) = reward;
        else
            q_next = predictQValues(dlnetTarget, nextState);
            q_values(action) = reward + gamma * max(q_next);
        end
        inputs(:,i) = state;
        targets(:,i) = q_values;
    end
    % Convert to dlarray
    dlX = dlarray(inputs, 'CB');
    dlY = dlarray(targets, 'CB');
    % Perform gradient descent
    [loss, gradients] = dlfeval(@modelGradients, dlnet, dlX, dlY);
    % Update the network parameters
    dlnet = dlupdate(@(w,g) w - alpha * g, dlnet, gradients);
end

function q_values = predictQValues(dlnet, state)
    % Predict Q-values for a given state
    dlX = dlarray(state, 'CB');
    dlY = predict(dlnet, dlX);
    q_values = extractdata(dlY);
end

function [loss, gradients] = modelGradients(dlnet, dlX, dlY)
    % Compute gradients and loss for training
    dlYPred = forward(dlnet, dlX);
    loss = mse(dlYPred, dlY);
    gradients = dlgradient(loss, dlnet.Learnables);
end

function [linkDelays, queueUtilizations] = updateNetworkConditions(N, neighbors, linkDistances)
    % Update network conditions with random delays and queue utilizations
    linkDelays = cell(N,1);
    queueUtilizations = rand(N,1); % Random queue utilization between 0 and 1
    for i = 1:N
        % Link delays depend on link distance and random factors
        linkDelays{i} = linkDistances{i} .* (1 + rand(length(neighbors{i}),1) .* queueUtilizations(i));
    end
end

function [path, totalDelay] = dijkstraRouting(source, destination, linkDelays, neighbors)
    % Implement Dijkstra's algorithm for shortest-path routing
    N = length(linkDelays);
    visited = false(N,1);
    distance = inf(N,1);
    previous = zeros(N,1);
    distance(source) = 0;
    queue = [source];
    while ~isempty(queue)
        [~, idx] = min(distance(queue));
        u = queue(idx);
        queue(idx) = [];
        if visited(u)
            continue;
        end
        visited(u) = true;
        if u == destination
            break;
        end
        for i = 1:length(neighbors{u})
            v = neighbors{u}(i);
            if ~visited(v)
                alt = distance(u) + linkDelays{u}(i);
                if alt < distance(v)
                    distance(v) = alt;
                    previous(v) = u;
                    queue = [queue, v];
                end
            end
        end
    end
    % Reconstruct path
    path = [];
    u = destination;
    while previous(u) ~= 0
        path = [u, path];
        u = previous(u);
    end
    if u == source
        path = [u, path];
    else
        path = [];
    end
    totalDelay = distance(destination);
end
 
