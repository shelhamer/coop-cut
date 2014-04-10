%%% example how to call the functions

% This is an example how to call the code for creating edge classes
% and unary potentials. For large images, it might be better to
% implement this in C/C++, as Matlab's k-means is not very efficient
% on this problem.
% That said, the file make_classes shows how an appropriate input file
% is generated.


% image
imagename = '../data/bee.jpg';

% user labels
userlabelimage = '../data/marks.png';

% 10 edge classes
num_classes = 10;

takelog = 0;
distance = 'sqEuclidean';

% also create unary potentials and save them
createUnaries = 1;

% create edge classes and save them
createEdgeClasses = 1;

imstub = 'bee';

make_classes(imagename, userlabelimage, num_classes, ...
    takelog, distance, createUnaries, createEdgeClasses, imstub);
