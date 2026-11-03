%% generate_range_cluster.m
%  Cluster-friendly variant of generate_training_data_multiangle.m.
%
%  Differences from the laptop version:
%    - Accepts an explicit list of phantom indices (idx_list) instead of
%      iterating 1:n_phantoms. Enables SLURM array parallelism.
%    - Skips outputs that already exist (resumable).
%    - Reads Field II location from environment variable FIELD_II_DIR
%      (if not given, it is taken from the FIELD_II_PATH environment variable).
%
%  This file is kept separate from generate_training_data_multiangle.m so
%  the laptop run is not perturbed.
%
%  Usage (typical sbatch task):
%      idx_list = i_start:i_end;
%      generate_range_cluster(output_dir, 'points', idx_list, angles_deg);
%
%  The reproducibility seeds, scatterer geometry and Field II calls are
%  identical bit-for-bit to generate_training_data_multiangle.m.

function generate_range_cluster(output_dir, phantom_type, idx_list, angles_deg)

    if nargin < 1, output_dir  = './field_ii_data_multiangle'; end
    if nargin < 2, phantom_type = 'points'; end
    if nargin < 3, idx_list    = 1:300; end
    if nargin < 4, angles_deg  = linspace(-16, 16, 75); end

    field_dir = getenv('FIELD_II_DIR');
    if isempty(field_dir)
        field_dir = getenv('FIELD_II_PATH');
        if isempty(field_dir)
            error('Set FIELD_II_PATH or pass field_dir as an argument.');
        end
    end
    addpath(field_dir);
    fprintf('Field II from: %s\n', field_dir);

    n_angles = length(angles_deg);

    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    sub_dir = fullfile(output_dir, phantom_type);
    if ~exist(sub_dir, 'dir'), mkdir(sub_dir); end

    %% ── Transducer parameters (must match generate_training_data_multiangle.m) ──
    c = 1540;
    fc = 5.208e6;
    fs = 20.832e6;

    n_elements     = 128;
    pitch          = 300e-6;
    element_width  = 270e-6;
    element_height = 5e-3;
    kerf           = pitch - element_width;
    focus_elevation = [0 0 20e-3];

    n_cycles = 2.5;
    t_pulse = 0 : 1/fs : n_cycles/fc;
    excitation = sin(2*pi*fc*t_pulse);

    frac_bw = 0.67;
    t_ir = -2/fc : 1/fs : 2/fc;
    impulse_response = sin(2*pi*fc*t_ir) .* ...
        exp(-pi^2 * (frac_bw * fc)^2 * t_ir.^2);

    x_range = 40e-3;
    z_min   = 5e-3;
    z_max   = 45e-3;

    %% ── Field II initialisation ──────────────────────────────────────
    field_init(0);
    set_field('c', c);
    set_field('fs', fs);
    set_field('use_rectangles', 1);

    tx = xdc_linear_array(n_elements, element_width, element_height, ...
                          kerf, 1, 1, focus_elevation);
    rx = xdc_linear_array(n_elements, element_width, element_height, ...
                          kerf, 1, 1, focus_elevation);

    xdc_excitation(tx, excitation);
    xdc_impulse(tx, impulse_response);
    xdc_impulse(rx, impulse_response);

    element_x = ((0:n_elements-1) - (n_elements-1)/2) * pitch;

    fprintf('Generating %s phantoms idx=[%d..%d] (count=%d) x %d angles...\n', ...
            phantom_type, idx_list(1), idx_list(end), length(idx_list), n_angles);

    %% ── Per-phantom loop ─────────────────────────────────────────────
    for k = 1:length(idx_list)
        i = idx_list(k);
        filename = fullfile(sub_dir, sprintf('%s_%04d.mat', phantom_type, i));

        if exist(filename, 'file') == 2
            fprintf('  [%d/%d] skip (exists): %s\n', k, length(idx_list), filename);
            continue;
        end

        tic;

        if strcmp(phantom_type, 'points')
            rng(i);
            [positions, amplitudes] = generate_point_phantom(x_range, z_min, z_max);
        elseif strcmp(phantom_type, 'cysts')
            rng(1000 + i);
            [positions, amplitudes] = generate_cyst_phantom(x_range, z_min, z_max);
        else
            error('Unknown phantom_type: %s', phantom_type);
        end

        rf_per_angle      = cell(n_angles, 1);
        t_start_per_angle = zeros(n_angles, 1);

        for a = 1:n_angles
            angle_rad = angles_deg(a) * pi / 180;
            delays = element_x * sin(angle_rad) / c;
            xdc_focus_times(tx, 0, delays);

            [rf_a, t_start_a] = calc_scat_multi(tx, rx, positions, amplitudes);
            rf_per_angle{a}      = rf_a;
            t_start_per_angle(a) = t_start_a;
        end

        max_D = 0;
        for a = 1:n_angles
            d_a = size(rf_per_angle{a}, 1);
            if d_a > max_D, max_D = d_a; end
        end

        rf_data = zeros(max_D, n_elements, n_angles, 'single');
        for a = 1:n_angles
            d_a = size(rf_per_angle{a}, 1);
            rf_data(1:d_a, :, a) = single(rf_per_angle{a});
        end

        t_start = t_start_per_angle;
        save(filename, ...
            'rf_data', 'angles_deg', 't_start', ...
            'positions', 'amplitudes', ...
            'fs', 'fc', 'c', 'n_elements', 'pitch', 'element_width', ...
            'phantom_type', '-v7');

        elapsed = toc;
        fprintf('  [%d/%d] %s  rf=(%d, %d, %d)  t=%.1fs\n', ...
                k, length(idx_list), filename, max_D, n_elements, n_angles, elapsed);
    end

    xdc_free(tx);
    xdc_free(rx);
    field_end;

    fprintf('Done. Saved %d phantoms to %s\n', length(idx_list), sub_dir);
end


%% ── Point-reflector phantom (identical to laptop version) ────────────
function [positions, amplitudes] = generate_point_phantom(x_range, z_min, z_max)
    n_points = 20;
    x = (rand(n_points, 1) - 0.5) * x_range;
    y = zeros(n_points, 1);
    z = z_min + rand(n_points, 1) * (z_max - z_min);
    positions  = [x, y, z];
    amplitudes = ones(n_points, 1);
end


%% ── Cyst phantom (identical to laptop version) ───────────────────────
function [positions, amplitudes] = generate_cyst_phantom(x_range, z_min, z_max)
    n_bg = 50000;
    x_bg = (rand(n_bg, 1) - 0.5) * x_range;
    y_bg = (rand(n_bg, 1) - 0.5) * 2e-3;
    z_bg = z_min + rand(n_bg, 1) * (z_max - z_min);
    amp_bg = randn(n_bg, 1);

    n_cysts    = randi([2, 4]);
    cyst_radii = 3e-3 + rand(n_cysts, 1) * 4e-3;
    cyst_x     = (rand(n_cysts, 1) - 0.5) * (x_range * 0.6);
    cyst_z     = z_min + 8e-3 + rand(n_cysts, 1) * (z_max - z_min - 16e-3);

    keep = true(n_bg, 1);
    for j = 1:n_cysts
        dist = sqrt((x_bg - cyst_x(j)).^2 + (z_bg - cyst_z(j)).^2);
        keep = keep & (dist > cyst_radii(j));
    end

    positions  = [x_bg(keep), y_bg(keep), z_bg(keep)];
    amplitudes = amp_bg(keep);
end
