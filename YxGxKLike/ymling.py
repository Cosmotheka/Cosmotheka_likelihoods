from make_yml import make_yml

fname_data = '../../data/1024_yxgxk_covmix/cls_cov_comb.fits'

for i in range(6):
    tname = f'LOWZ__{i}'
    make_yml(params_vary=[f'ygk_{tname}_lMmin_0',
                          f'ygk_{tname}_lM1_0',
                          'ygk_mass_bias',
                          'ygk_rhogy'],
             corrs=[(tname, tname),
                    (tname, 'YMILCA')],
             bias_model='HaloModel',
             kmax=1.0,
             fname_data=fname_data,
             dirname_out=f'yxg_{i}',
             sampler='minimize', debug=False)
    make_yml(params_vary=[f'ygk_{tname}_lMmin_0',
                          f'ygk_{tname}_lM1_0',
                          'ygk_rhogm',
                          'sigma8'],
             corrs=[(tname, tname),
                    (tname, 'KAPPA')],
             bias_model='HaloModel',
             kmax=1.0,
             fname_data=fname_data,
             dirname_out=f'gxk_{i}',
             sampler='minimize', debug=False)
    make_yml(params_vary=[f'ygk_{tname}_lMmin_0',
                          f'ygk_{tname}_lM1_0',
                          'ygk_mass_bias',
                          'ygk_rhogy',
                          'sigma8'],
             corrs=[(tname, tname),
                    (tname, 'YMILCA'),
                    (tname, 'KAPPA')],
             bias_model='HaloModel',
             kmax=1.0,
             fname_data=fname_data,
             dirname_out=f'yxgxk_{i}',
             sampler='minimize', debug=False)

pvary = [f'ygk_LOWZ__{i}_b0' for i in range(6)]
pvary += [f'ygk_LOWZ__{i}_wz' for i in range(6)]
corrs = []
for i in range(6):
    for j in range(i, 6):
        corrs.append((f'LOWZ__{i}', f'LOWZ__{j}'))
make_yml(params_vary=pvary,
         corrs=corrs,
         bias_model='Linear',
         kmax=0.2,
         fname_data=fname_data,
         dirname_out=f'xcorr_gc',
         sampler='minimize')
