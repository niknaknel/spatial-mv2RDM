# Spatial m-v2RDM
This repository contains code relating to my Master's thesis. The thesis document
is uploaded here.

The full implementation can be found in the `src` folder.

### _Abstract_
Reduced density matrices (RDMs) offer a more scalable alternative to full
wavefunctions when performing chemical calculations. The variational two-electron
RDM (v2RDM) method exploits the efficiency of RDMs, employing
semidefinite programming (SDP) to enable polynomial scaling of ground
state simulations. Recent work by Avdic & Mazziotti seeks to improve the
performance of the v2RDM by incorporating classical shadow constraints,
simultaneously reducing the number of measurements required for tomography.
Drawing from this work, we introduce a spatial orbital variant of the v2RDM
with measurement constraints (m-v2RDM). The proposed method achieves
comparable accuracy for small to medium-sized molecules such as H2, H4,
and HF, while substantially reducing memory and runtime costs. Its
comparatively simple implementation also allows for the approximation
of larger systems like N2, which are otherwise intractable on modest
computational resources using standard v2RDM. As a pedagogical resource,
the spatial variant more closely resembles the underlying theory, making 
it an accessible introduction to RDMs. The spatial m-v2RDM further highlights 
the complementary nature of measurement constraints and N-representability
conditions, framing the RDM as a potential tool for noise mitigation in 
quantum information processing.

## Example usage
The `notebooks/example.ipynb` file contains some examples on how
to run the spatial m-v2RDM for H$_2$ and H$_4$.

## Mappings demo
The `notebooks/mappings-demo.ipynb` file demonstrates the correctness of the
Q and G mappings in `src/dqg.py`, i.e. the $N$-representability conditions used
to constrain the SDP.

## True classical shadows
The `notebooks/classical-shadows.ipynb` file contains some code attempting to
implement the suggestion for true classical shadows in Appendix C of my thesis.