# Latent-Semantic-Analysis-MATLAB

% Define the Term-Document Matrix
A = [
    0 1 1 0 1;  % equations
    0 1 1 0 0;  % algebra
    1 0 1 0 0;  % calculus
    1 0 0 0 1;  % change
    0 0 1 0 1;  % differential
    0 1 1 0 0;  % solving
    0 0 0 1 0;  % angles
    1 0 0 0 0;  % derivatives
    0 0 0 1 0;  % geometry
    1 0 0 0 0;  % integrals
    0 0 0 0 1;  % modeling
    0 0 0 0 1;  % physical
    1 0 0 0 0;  % rates
    0 0 0 1 0;  % relationships
    0 0 0 1 0;  % shapes
    0 0 0 1 0;  % spatial
    0 1 0 0 1   % systems
];
% Perform Singular Value Decomposition
[U, S, V] = svd(A);
disp(U);
disp(S);
disp(V);

% Reduce the rank
r = 2; % Using the top 2 singular values

U_reduced = U(:, 1:r);
S_reduced = S(1:r, 1:r);
V_reduced = V(:, 1:r);

% Compute projections
term_projections = U_reduced * S_reduced;  % Terms onto concepts
document_projections = S_reduced * V_reduced'; % Documents onto concepts

% Display projections
disp('Term-to-Concept Projections:');
disp(term_projections);
disp('Document-to-Concept Projections:');
disp(document_projections');

terms = {'equations', 'algebra', 'calculus', 'change', 'differential', ...
         'solving', 'angles', 'derivatives', 'geometry', 'integrals', ...
         'modeling', 'physical', 'rates', 'relationships', 'shapes', ...
         'spatial', 'systems'};

first_concept_projections = term_projections(:, 1); % First column of term projections

% Create a bar plot
figure;
bar(first_concept_projections, 'b');
set(gca, 'XTick', 1:length(terms), 'XTickLabel', terms, 'FontSize', 10);
xtickangle(45); % Rotate term labels for better readability
ylabel('Projection onto First Concept', 'FontSize', 12, 'FontWeight', 'bold');
title('Term Projections onto First Concept', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

hold on;
threshold = 0.5; % Define a threshold for significant projections
significant_indices = abs(first_concept_projections) > threshold;
bar(find(significant_indices), first_concept_projections(significant_indices), 'r');
hold off;

% Define the queries and their corresponding term indices
queries = {
    'Calculus', [1, 5, 9];         % Calculus, Derivatives, Integrals
    'Algebra', [2, 6, 16];         % Algebra, Equations, Variables
    'Geometry', [8, 11, 13]        % Geometry, Shapes, Spatial
};

% Initialize results
results = struct();

% Loop through each query
for q = 1:length(queries)
    query_name = queries{q, 1};
    query_indices = queries{q, 2};
    
    % Compute the query vector (average of selected term vectors)
    query_vector = mean(term_projections(query_indices, :), 1);
    
    % Compute cosine similarity between query vector and each document vector
    similarities = zeros(size(document_projections, 1), 1);
    for i = 1:size(document_projections, 1)
        doc_vector = document_projections(i, :);
        similarities(i) = dot(query_vector, doc_vector) / (norm(query_vector) * norm(doc_vector));
    end
    
    % Store results
    results(q).Query = query_name;
    results(q).QueryVector = query_vector;
    results(q).Similarities = similarities;
end

% Display results
for q = 1:length(results)
    disp(['Query: ', results(q).Query]);
    disp('Query Vector:');
    disp(results(q).QueryVector);
    disp('Cosine Similarities with Documents:');
    disp(results(q).Similarities);
end
