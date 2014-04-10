function [] = make_classes(imagename, userlabelimage, num_classes, ...
    takelog, distance, createUnaries, createEdgeClasses, imstub)
%
% imagename:    path to your image
% userlabelimage: path to image with user labels
% num_classes:  number of edge classes/types
% takelog:      if yes (one/true), then log ratios are used as features
%               instead of linear gradients
% distance:     distance='cityblock' uses L1, distance='sqEuclidean' uses 
%               squared L2
% createUnaries: create unary potentials
% createEdgeClasses: create edge classes
% imstub:       prefix of files to be saved (e.g. if it is myIm, the edge 
%               classes file will be myImcl10.bin for 10 edge classes
%
% This code is example code to compute unary potentials (here via
% histograms; one can also do GMMs) and edge classes needed to do image
% segmentation with cooperative cuts. It saves a file with unary potentials
% and one with edge classes.
% If log ratios are used, this only computes edge classes and not unary
% potentials.
%
% This needs the Matlab statistics toolbox.
%
% Code by Stefanie Jegelka and Jeff Bilmes
% Please acknowledge the CVPR 2011 paper "Submodularity beyong submodular
% energies: coupling edges in graph cuts" when using the code.

% just do some variable name truncations
labname = userlabelimage;
imname = imagename;


% read in image
A = double(imread(imname));


n1 = size(A,1);
n2 = size(A,2);
n = n1*n2;


Av = reshape(double(A), [], 3);

if takelog
    display('log version');
    Av = log(max(Av,eps));
    createUnaries = 0;
    fprintf('unary potentials need to be done with takelog=0 !\n');
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% create edge classes

    
% We start by taking differences of adjacent pixels. The order here is the
% same that is used in the C++ code provided online, it is important that
% the order of the entries is retained.
% This is done in four blocks (directions).

disp('creating edges ...');

% vertical
% create elist-part and use that to compute gradients
tmpli = [];
elist = [];
for i=1:n2
    tmpli = [tmpli; ...
        [([1:(n1-1)]'+(i-1)*n1), ([2:n1]'+(i-1)*n1)] ];
end
wts = Av(tmpli(:,1),:) - Av(tmpli(:,2),:);
elist = [elist; tmpli];


% horizontal, to right and to left
tmpli = [];
for i=1:(n2-1)
    tmpli = [tmpli; ...
        [([1:n1]'+i*n1), ([1:n1]'+(i-1)*n1)] ];
end
wts1 = Av(tmpli(:,1),:) - Av(tmpli(:,2),:);
wts = [wts; [wts1]];
elist = [elist; tmpli];

diagstart = size(elist,1)+1;
% diagonal \
tmpli = [];
for i=1:(n2-1)
    tmpli = [tmpli; ...
        [([1:(n1-1)]'+(i-1)*n1), ([2:n1]'+(i)*n1)] ];
end
wts1 = Av(tmpli(:,1),:) - Av(tmpli(:,2),:);
wts = [wts; [wts1]];
elist = [elist; tmpli];


% diagonal /
tmpli = [];
for i=1:(n2-1)
    tmpli = [tmpli; ...
        [([2:n1]'+(i-1)*n1), ([1:(n1-1)]'+i*n1)] ];
end
wts1 = Av(tmpli(:,1),:) - Av(tmpli(:,2),:);
wts = [wts; wts1];
elist = [elist; tmpli]; 


    
% now we are done creating the vectors of pixel differences
% next we transform it to the standard (Gaussian) edge weights
tmpli = sum( wts.^2, 2);
tmpw = reshape([tmpli'; tmpli'], [], 1);
nullinds = (tmpw < eps);

fprintf('\n%d zero differences\n', sum(nullinds));

sigma = mean(tmpli);
wtsfull = wts;
weights = 0.05+0.95*exp( - tmpli / (2*sigma));
    
mw = max(weights);
m = size(elist,1);
clear tmpw;
    

% cluster the difference vectors into num_classes classes; the ones where
% the difference is zero will go in an extra class

if createEdgeClasses

    disp('clustering edges ...');

    % double each entry in wts, in negative form to match tmpw
    wts = [wtsfull, -wtsfull]';
    wts = reshape(wts, 3, [])';
    clear wtsfull; clear elist;
        
    classes = ones(2*m,1)*num_classes;
            
            sname = sprintf('%scl%d.bin',imstub, num_classes);
            
            % make edge classes by clustering (k-means)
            str1 = RandStream.create('mrg32k3a', 'Seed', 27, 'NumStreams',1);
            RandStream.setDefaultStream(str1);
            % if ~exist(sprintf('%s.gz',sname),'file')
            
            % k-means gives vector if assignments
            restcl = kmeans(wts((nullinds==0),:), num_classes, 'EmptyAction', 'singleton','Distance',distance);            
            classes(nullinds==0) = restcl-1;
            
            %%%%%%%%% WRITE EDGE CLASSES INTO A FILE %%%%%%%%%%%%%%%%%
            fp = fopen(sname,'wb');
            fwrite(fp, num_classes, 'int32');
            fwrite(fp, classes, 'int32');
            fclose(fp);
            % use the following if you want to gzip the file (need to
            % unzip it to use it)
            % system( sprintf('gzip %s', sname));
                        
            for i2=min(classes):max(classes)
                fprintf('class %d: %d edges\n', i2, sum(classes==i2));
            end
            % end
            
end

clear elist;
clear wts;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% create unary potentials %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if createUnaries
    
    disp('creating unary potentials');
    
    % read in the label file, coded by red and blue pixels
    L = double(imread( labname));
    if numel(L(:,:,1)) ~= size(Av,1)
        fprintf('WARNING: sizes do not match! label file %s has different size than image\n', labname);
    end
    backgPix = find( (L(:,:,3)>0) .* (L(:,:,1)==0));
    objPix = find((L(:,:,1)>0) .* (L(:,:,3)==0));
    clear L;
    
    sname = sprintf('%subinU.bin',imstub);
    %    if ~exist(sprintf('%s.gz',sname),'file') || overwrite
        
    binsz = 4;
    % the next function returns two weight vectors, one for the source and
    % one for the sink edges
    [wo,wb] = make3Dhist(binsz, Av, objPix, backgPix);
    wsink = -wo;
    wsource = -wb;   
        
    % the pixels that the used marked receive heavy weigths
    hardwt = (sum(weights) + sum(wsink))/2;
    wsource(objPix) = hardwt;
    wsink(backgPix) = hardwt;
    
    wsource = reshape(reshape(wsource, n1, n2)',[],1);
    wsink = reshape(reshape(wsink, n1, n2)',[],1);
        
    fp = fopen(sname,'wb');
    fwrite(fp, wsource, 'double');
    fwrite(fp, wsink, 'double');
    fclose(fp);
    % if you want to gzip the file then use this:
    % system( sprintf('gzip %s', sname));
    
        %fprintf('statistics: n=%d, m=%d. Min src weight: %1.5f, max: %1.5f, weights: %1.5f - %1.5f\n.', ...
        %    n, m, min(wsource), max(wsource), 50*min(weights), 50*max(weights));
        
    % end

        
    %%%%%%%%%%%%%%% the GMM unary file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if 0 %%%% put to 1 if you have downloaded the GMM-GMR-v2.0 code
	% from www.calinon.ch/sourcecodes.php
        % (adjust then the paths in the file makeGMM.m)
        
        sname = sprintf('%sugmmU.bin', imstub);
        %if ~exist(sprintf('%s.gz',sname),'file') || overwrite
        
        % k-means unaries
        [wsource,wsink] = makeGMM(A, backgPix, objPix);
        hardwt = (sum(weights) + sum(wsource) + sum(wsink))/3;
        wsource(objPix) = hardwt;
        wsink(backgPix) = hardwt;
        wsource = reshape(reshape(wsource, n1, n2)',[],1);
        wsink = reshape(reshape(wsink, n1, n2)',[],1);
        
        
        fp = fopen(sname,'wb');
        fwrite(fp, wsource, 'double');
        fwrite(fp, wsink, 'double');
        fclose(fp);
        
        % use this if you want to zip the file:
        % system( sprintf('gzip %s', sname));
        
        %end
        
    end
    
    clear wsource;
    clear wsink;
    
end



