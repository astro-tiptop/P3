close all;
clear all;

%% MANAGE WORKING PATHS
path_oomao = '/home/omartin/Projects/SIMULATIONS/OOMAO/mfiles_old/';
path_workspace = '/home/omartin/Projects/PSFR/GEMS_PSFR/CODES/';
addpath(genpath(path_oomao),path_workspace);

%% MANAGE DATA PATHS
path_data = '/run/media/omartin/OlivierMartinHDD/DATA/GEMS_DATA/TR14/';
path_K = 'Ks/';
path_H  = 'Hband/';
path_J   = 'Jband/';
path_Br = 'brgamma/';

%% GET DATA IDENTITY
dataID  = cell(1,4);
for c=1:4
    switch c
        case 1
            path = path_K;
        case 2
            path = path_H;
        case 3
            path = path_J;
        case 4
            path = path_Br;
    end
    cd([path_data,path]);
    list     = ls();
    fname    = textscan( list, '%s');
    fname    = fname{1};
    nObj     = size(fname,1);
    idxFits   = false(1,nObj);
    for k=1:nObj
        idxFits(k) = any(strfind(fname{k},'proj.fits'));
    end
    idK = find(idxFits);
    nImg = nnz(idxFits);
    dataID{c} = strings(1,nImg);
    for k=1:nImg
        dataID{c}(k) = fname{idK(k)};
    end
end

%%
%img_K = fitsread([path_data,path_K,cell2mat(dataID{1}(1))],'Image');
%img_K = img_K.*(img_K>0);
img_H = fitsread([path_data,path_H,cell2mat(dataID{2}(1))],'Image');
img_H = img_H.*(img_H>0);
%2644.78 148.23 9.900 0.001
%img_J = fitsread([path_data,path_J,cell2mat(dataID{3}(1))],'Image');
%img_J = img_J.*(img_J>0);
%%
close all;
h = figure;
imagesc(log10(img_K),[3.5,5]);
set(gca,'XTick',[],'Ytick',[]);
colormap('hot');
pbaspect(h.CurrentAxes,[1,1,1]);

h = figure;
imagesc(log10(img_H),[3,5]);
set(gca,'XTick',[],'Ytick',[]);
colormap('autumn');
pbaspect(h.CurrentAxes,[1,1,1]);

h = figure;
imagesc(log10(img_J),[1,5]);
set(gca,'XTick',[],'Ytick',[]);
colormap('winter');
pbaspect(h.CurrentAxes,[1,1,1]);

%% LOAD DATA
ref_K = [2119 2904];
ref_H = [2045 3003];
ref_J  = [2119 2903];
ref_Br  = [2120 2904];
ref     = [ref_K;ref_H;ref_J;ref_Br];
nEdg  = 200;
nRes  = 128;

for c=1:1
      switch c
            case 1
                path = path_K;
                x_ref = [347 4020 1972   3697  3018]; 
                y_ref = [832 1007 2150   1422  3575];
            case 2
                path = path_H;
                x_ref = [4244 3632 806 2068 3257]; 
                y_ref = [606 864 3524 4496 3809];
            case 3
                path = path_J;
                x_ref = [4020 752 854 2525 4383]; 
                y_ref = [1007 2186 4480 3942 3983];
            case 4
                path = path_Br;
                x_ref = [] ;
                y_ref = [];
      end
        
    for k=2:2%numel(dataID{c})
                                     
        % Grab information from the header
        path_img = [path_data,path,cell2mat(dataID{c}(k))];
        hdr = fitsinfo(path_img);
        hdr  = hdr.PrimaryData.Keywords;
        
        list  = hdr(:,1);
        val  = hdr(:,2);
        % Observing conditions
        wvl  = cell2mat(val(strcmp(list,'WAVELENG')))*1e-10; %in meter
        Samp = constants.radian2mas*wvl/8.1/20/2;
        airm  = cell2mat(val(strcmp(list,'AIRMASS')));
        r0  = cell2mat(val(strcmp(list,'RZEROVAL')))*(wvl/500e-9)^(1.2)*airm^(-3/5);
        sr  = cell2mat(val(strcmp(list,'LGSSTRHL')));
        
%         % NGS config
%         ngs_ra = zeros(1,3);
%         ngs_dec = zeros(1,3);
%         ngs_wav= zeros(1,3);
%         for i=1:3
%             ngs_ra(i) = cell2mat(val(strcmp(list,['GWFS',num2str(i),'RA'])));
%             ngs_dec(i) = cell2mat(val(strcmp(list,['GWFS',num2str(i),'DEC'])));
%             ngs_wav(i) = cell2mat(val(strcmp(list,['GWFS',num2str(i),'WAV'])))*1e-10; %in meter
%         end
%         lgs_wav = 589e-9;
        % Detector
%         gains = [2.434,2.01,2.411,2.644]; %in e-/ADU
        
%         lnrs = cell2mat(val(strcmp(list,'LNRS')));
%         switch lnrs
%             case 2
%                 ron_arr = [26.53,19.1,27.24,32.26];
%             case 8
%                 ron_arr = [23.63,9.85,14.22,16.78];
%             case 16
%                 ron_arr = [10.22,7.44,10.61,12.79];
%             otherwise
%                 ron_arr = 0;
%         end
%         ron = median(ron_arr./gains);
        % Get the image
        img = fitsread(path_img,'Image');
        img = img.*(img>0);
                                               
        
        nP     = length(x_ref);
        im_sub = zeros(nP,nRes,nRes);
                                        
        % Get PSFS
        
        for j=1:nP
            % Getting a sub-field
            idx  = x_ref(j)-nEdg/2+1:x_ref(j)+nEdg/2;
            idy  = y_ref(j)-nEdg/2+1:y_ref(j)+nEdg/2;            
            im_sub(j,:,:) = tools.processImage(img(idx,idy),0,1,0,Samp,'fovInPixel',nRes,'masking',false,'rebin',0,'tfccd',false,'thresholding',-Inf);
        end
        
        
        switch c
            case 1
                fitswrite(im_sub,['/home/omartin/Projects/PSFR/GEMS_PSFR/DOCUMENTS/psf_gems_K_r0_',num2str(r0),'cm_sr_',num2str(sr),'_Samp_',num2str(Samp),'_',cell2mat(dataID{c}(k))])
                
            case 2
                fitswrite(im_sub,['/home/omartin/Projects/PSFR/GEMS_PSFR/DOCUMENTS/psf_gems_H_r0_',num2str(r0),'cm_sr_',num2str(sr),'_Samp_',num2str(Samp),'_',cell2mat(dataID{c}(k))])
                
            case 3
                fitswrite(im_sub,['/home/omartin/Projects/PSFR/GEMS_PSFR/DOCUMENTS/psf_gems_J_r0_',num2str(r0),'cm_sr_',num2str(sr),'_Samp_',num2str(Samp),'_',cell2mat(dataID{c}(k))])
                
            case 4
                fitswrite(im_sub,['/home/omartin/Projects/PSFR/GEMS_PSFR/DOCUMENTS/psf_gems_Br_r0_',num2str(r0),'cm_sr_',num2str(sr),'_Samp_',num2str(Samp),'_',cell2mat(dataID{c}(k))])                
        end
    end
end

