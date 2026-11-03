%% run_cluster_generation.m
%  Entry point for SLURM array tasks.
%
%  Reads from env vars:
%    PHANTOM_TYPE   : 'points' or 'cysts'
%    IDX_START      : starting phantom index (1-based, inclusive)
%    IDX_END        : ending phantom index (inclusive)
%    OUTPUT_DIR     : output dir, default './field_ii_data_multiangle'
%    N_ANGLES       : number of angles, default 75 (linspace -16..16)
%    FIELD_II_DIR   : Field II install path (forwarded to generate_range_cluster)
%
%  Usage in sbatch (after `cd "datasets/Field II"`):
%    matlab -batch "run_cluster_generation"

phantom_type = getenv('PHANTOM_TYPE');
if isempty(phantom_type), phantom_type = 'points'; end

idx_start = str2double(getenv('IDX_START'));
idx_end   = str2double(getenv('IDX_END'));
if isnan(idx_start), idx_start = 1; end
if isnan(idx_end),   idx_end   = 1; end

output_dir = getenv('OUTPUT_DIR');
if isempty(output_dir), output_dir = './field_ii_data_multiangle'; end

n_angles_str = getenv('N_ANGLES');
if isempty(n_angles_str)
    n_angles = 75;
else
    n_angles = str2double(n_angles_str);
    if isnan(n_angles), n_angles = 75; end
end

angles_deg = linspace(-16, 16, n_angles);
idx_list = idx_start:idx_end;

fprintf('=== run_cluster_generation ===\n');
fprintf('  phantom_type : %s\n', phantom_type);
fprintf('  idx_list     : [%d..%d] (count=%d)\n', idx_start, idx_end, length(idx_list));
fprintf('  output_dir   : %s\n', output_dir);
fprintf('  n_angles     : %d (linspace -16..16)\n', n_angles);
fprintf('  hostname     : %s\n', char(java.net.InetAddress.getLocalHost.getHostName));
fprintf('  date         : %s\n', datestr(now, 31));
fprintf('================================\n');

addpath(pwd);  % so generate_range_cluster.m is on the path
generate_range_cluster(output_dir, phantom_type, idx_list, angles_deg);

fprintf('=== run_cluster_generation done %s ===\n', datestr(now, 31));
