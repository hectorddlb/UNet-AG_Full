%% run_generation_multiangle.m
%  Generates the multi-angle training dataset.
%
%  RUN ON LAPTOP — the El-Insurgente cluster has no MATLAB installation.
%
%  Steps:
%    1. cd "datasets/Field II"
%    2. matlab -batch "run_generation_multiangle"
%       (or open MATLAB and >> run_generation_multiangle)
%    3. After completion, rsync to the cluster (Git Bash, from the
%       laptop's thesis directory):
%         rsync -avzP -e "ssh -p 2284" \
%           "datasets/Field II/field_ii_data_multiangle/" \
%           <user>@<cluster-host>:<destination-path>/field_ii_data_multiangle/
%
%  Output structure:
%    field_ii_data_multiangle/
%      points/points_0001.mat ... points_0300.mat
%      cysts/cysts_0001.mat   ... cysts_0300.mat
%
%  Each .mat (per phantom):
%    rf_data    : (max_D, 128, 75) single  — zero-padded per angle
%    angles_deg : (75,) double             — linspace(-16, 16, 75)
%    t_start    : (75,) double             — per-angle start time [s]
%    positions, amplitudes, fs, fc, c, n_elements, pitch, element_width, phantom_type
%
%  Reproducibility: rng(i) per phantom; type-disjoint seeds (points: i,
%  cysts: 1000+i). Re-running produces identical scatterer geometries.
%
%  Cost estimate (single laptop MATLAB instance, CPU only):
%    Cysts dominate (50,000 scatterers each); points are ~100x faster.
%    Plan for 2-7 days total wall-clock. Stagger by running points first.
%
%  Recovery: if MATLAB crashes mid-run, simply re-launch — already-saved
%  .mat files are skipped only if you wrap the loop with an exists() check
%  (not done here for simplicity; deleting the partial .mat is enough).

output_dir = './field_ii_data_multiangle';
angles_deg = linspace(-16, 16, 75);

fprintf('============================================\n');
fprintf(' Field II Multi-Angle Generation\n');
fprintf(' Phase 2v3 Plan B - 600 phantoms x %d angles\n', length(angles_deg));
fprintf(' Output: %s\n', output_dir);
fprintf('============================================\n\n');

fprintf('--- Points (300 phantoms) ---\n');
generate_training_data_multiangle(output_dir, 'points', 300, angles_deg);

fprintf('\n--- Cysts (300 phantoms) ---\n');
generate_training_data_multiangle(output_dir, 'cysts', 300, angles_deg);

fprintf('\n============================================\n');
fprintf(' Complete: 600 phantoms x %d angles\n', length(angles_deg));
fprintf('============================================\n');
