clear all;close all;
clc;
path_data_gems = '/home/omartin/Projects/PSFR/GEMS_PSFR/DOCUMENTS/';
path_oomao = '/home/omartin/Projects/SIMULATIONS/OOMAO/mfiles_old/';
addpath(genpath(path_oomao));

%% LOAD GSAOI PSF

psf_gsaoi_K = fitsread([path_data_gems,'psf_gems_K_r0_54.914cm_sr_2_Samp_1.3687_rgS20190416S0115_proj.fits']);
psf_gsaoi_H = fitsread([path_data_gems,'psf_gems_H_r0_40.7014cm_sr_3_Samp_1.0409_rgS20190418S0042_proj.fits']);
psf_gsaoi_J = fitsread([path_data_gems,'psf_gems_J_r0_43.4147cm_sr_8_Samp_0.79577_rgS20190417S0096_proj.fits']);

psf_fit_K      = permute(fitsread([path_data_gems,'psf_fit_K.fits']),[3,2,1]);
psf_fit_H      = permute(fitsread([path_data_gems,'psf_fit_H.fits']),[3,2,1]);
psf_fit_J      = permute(fitsread([path_data_gems,'psf_fit_J.fits']),[3,2,1]);

x_fit_K        = fitsread([path_data_gems,'param_fit_K.fits']);
x_fit_H        = fitsread([path_data_gems,'param_fit_H.fits']);
x_fit_J        = fitsread([path_data_gems,'param_fit_J.fits']);

nPx = size(psf_gsaoi_J,2);
nImg = size(psf_gsaoi_J,3);

%% PLOT
fontsize=16;
close all;
xx = linspace(-1,1,nPx)*20;

figure;
pg = (squeeze(median(psf_gsaoi_K(:,end/2+1,:),1))');
pf = squeeze(mean(psf_fit_K(:,end/2+1,:),1))';
semilogy(xx,pg,'r');hold on;
semilogy(xx,pf,'b');
semilogy(xx,abs(pf-pg),'k');
xlabel('Angular separation (mas)','interpreter','latex','fontsize',fontsize);
set(gca,'FontSize',fontsize,'FontName','cmr12','TickLabelInterpreter','latex');
legend({'GSAOI PSF','Model adjusted','Residual'},'interpreter','latex','fontsize',fontsize);
title('K-band filter','interpreter','latex','fontsize',fontsize);
pbaspect([1.6,1,1]);

figure;
img = abs(reshape(permute(psf_gsaoi_K([1,2,4,5],:,:),[2,3,1]),nPx,4*nPx));
imf = abs(reshape(permute(psf_fit_K([1,2,4,5],:,:),[2,3,1]),nPx,4*nPx));
imagesc(log10( [img;imf;abs(imf-img)]),[-1,5]);
set(gca,'XTick',[],'YTick',[]);
pbaspect([4,3,1]);
cb = colorbar();
cb.TickLabelInterpreter = 'latex';
cb.FontSize = 14;
