"""Python replica of DRUS (Zhang et al. 2023, DDRM for US reconstruction).

Port of the MATLAB scripts of the official repo
https://github.com/Yuxin-Zhang-Jasmine/DRUS-v1 under the conventions of the
unet_ag pipeline. The public API is in ``drus.inference`` (class ``DRUS``) and
the command-line tool in ``drus.cli`` (``drus``); the artifacts it needs are
built with ``drus.preparation``.
"""
