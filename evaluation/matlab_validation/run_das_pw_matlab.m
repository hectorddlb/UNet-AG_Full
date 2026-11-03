function run_das_pw_matlab(fnum_kind, angle_ids, out_file)
% Reference implementation das_pw.m (mschiffn/f_number, commit ef85e74) on the
% PICMUS experimental contrast scenario, on CPU.
%   fnum_kind : 'fdf' (f_numbers.grating.angle_lb(60, 1.5, -1)) or 'das' (constant(0))
%   angle_ids : MATLAB (1-based) indices of the angles; the central one (0°) is 38
%   out_file  : output .mat with each complex image (v7.3)
% Requires the input of prepare_input.py in work/ and the f_number repository
% at the path given by the F_NUMBER_DIR environment variable.
here = fileparts(mfilename('fullpath'));
fdir = getenv('F_NUMBER_DIR');
if isempty(fdir), fdir = fullfile(here, 'work', 'f_number'); end
addpath(fdir);
S = load(fullfile(here, 'work', 'exp_contrast_rf.mat'));
if strcmp(fnum_kind, 'fdf'), F = f_numbers.grating.angle_lb(60, 1.5, -1); else, F = f_numbers.constant(0); end
imgs = complex(zeros(numel(S.positions_z), numel(S.positions_x), numel(angle_ids)));
secs = zeros(1, numel(angle_ids));
for i = 1:numel(angle_ids)
    a = angle_ids(i); t = tic;
    imgs(:,:,i) = das_pw(S.positions_x(:).', S.positions_z(:).', double(S.data_RF(:,:,a)), S.fs, S.angles(a), ...
        S.width, S.pitch, S.c0, [2.25e6, 6.75e6], 0, windows.tukey(0.2), F, normalizations.on, platforms.cpu);
    secs(i) = toc(t); fprintf('%s angle %d: %.1f s\n', fnum_kind, a, secs(i));
end
save(out_file, 'imgs', 'angle_ids', 'secs', '-v7.3');
end
