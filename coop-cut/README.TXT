////////////////////////////////////////////////////////////
//         Partial code for Cooperative Cuts              //
////////////////////////////////////////////////////////////

This code implements parts of the paper

S. Jegelka and J. Bilmes. "Submodularity beyond submodular energies: coupling
edges in graph cuts". IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2011.

If you use this code, then cite the above paper. The paper contains
details about the algorithm and the model.
The code uses the maxflow code from
http://www.cs.ucl.ac.uk/staff/V.Kolmogorov/software.html.
Please downlowad the code from this link, or from
http://vision.csd.uwo.ca/code/maxflow-v3.01.zip.
In your directory for the Cooperative Cut code, make a subirectory
"maxflow", and put the graph cut code there, without any further
subdirectories. You will also need OpenCV for some functions (only for
I/O). Note that the Maxflow code comes with its own license.

This part of the code does not get the edge classes -- you will have
to derive them yourself, depending on your data. In the above paper,
we clustered graph edges by their feature vectors (using k-means).


LICENSE & DISCLAIMER
---------------------

    Copyright 2011 Stefanie Jegelka (jegelka@tuebingen.mpg.de) and
    Jeff Bilmes (bilmes@ee.washington.edu)

    This software can be used for research purposes for free. In any
    resulting publication, you should cite the above CVPR paper. For
    commercial use, please contact the authors above.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
    CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
    THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    SUCH DAMAGE.


COMPILING AND USING THE CODE
-----------------------------

The file call.cpp shows how to use the code.

To compile the code, type "make". The code was tested on 64-bit Ubuntu.
You will also need to install OpenCV, but this is only used to read
the images and show the segmentation in call.cpp, if you input the
image in a different fashion, then you do not need OpenCV.

The code here implements the following. Given an image and classes of
its edges (e.g., obtained from k-means clustering of edge feature
vectors), we try to find the minimum cooperative cut

min_C w(C \inter E_terminal) + gamma* f(C \inter E_interpixel),

where f is a nondecreasing submodular function. Costfun.cpp uses

f(C) = \sum_i f_i(C \inter S_i)

for edge groups S_i, and
f_i(A) = w(A) if  w(A) <= theta*w(S_i); and f_i(A) = sqrt(w(A)-theta*w(S_i)+1)-1 + theta*w(S_i).

Try calling
./callcc testdata/005_bee.jpg testdata/005_beecl15.bin testdata/005_beeugmmU.bin 1.0 0.003


The components are:

Edge_data:
- generic edge information class that contains
  m: the number of edges
  height: image height
  width: image width
  weights: double array of length m, one entry for each pair (u,v) and (v,u) of edges
  classes: int array of length 2*m, one entry for each edge
- you can also write your own Edge_data

Edat_read:
- inherits from Edge_data, this version reads in files.

Costfun:
- cost function class that holds teh graph, all edge weights and the cost and can
  update the edge weights.

itBM:
- iterative minimization procedure; uses a Costfun object.
