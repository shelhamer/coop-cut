function [wo,wb] = make3Dhist(binsz, A, obj, backg)
%
% A: n x 3 vector
% binsz: number of pixels in a bin (1D)
% obj: marked object pixels
% backg: marked background pixels
% pixel range: 0 to 255

%%%%%%% this generates unary potentials from histograms

bins = ((binsz-1)/2):binsz:255;
nbtot = length(bins)^3;
bmax = floor(255/binsz)+1;

Ab = floor(A/binsz); % bin for each dimension
clear A;
coded = Ab(:,1)*bmax^2 + Ab(:,2)*bmax + Ab(:,1); % bin numbers


epsi = 1/(4*numel(coded));


[cto] = hist(coded(obj), 0:(nbtot-1)) + epsi;

[ctb] = hist(coded(backg), 0:(nbtot-1)) + epsi;


p_obj = log(cto) - log(sum(cto));
p_back = log(ctb) - log(sum(ctb));

size(p_obj)
size(p_back)


wo = p_obj(coded+1);
wb = p_back(coded+1);
