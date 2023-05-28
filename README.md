# Pattern Removal filter
This script implements a technique for removal of interference patterns and periodic noise from images.

## Background
This algorithm was developed for images of the two transmitted beams and the two beams generated in a four-wave mixing (FWM) process. In our laser beam images, we consistently observed interference patterns and periodic noise. It was crucial to eliminate these interference patterns and periodic noise before conducting any analyses.

## Algorithm
This algorithm is based on Cannon [1]. It automates the proccess of identifying and eliminating the frequencies associated with the periodic noise in the Fourier transform of the images. It employs Welch's averaging method [2] to derive a modified power spectrum, enabling the identification of Fourier transform components associated with periodic noise. Subsequently, a notch filter is created based on these components and applied to the image for effective elimination.

In the figure bellow we display the original image (a), the power spectrum (b), and the modified power
spectrum (c). The modified power spectrum allows us to clearly observe the other components of
the Fourier transform.

<img src="https://github.com/PatrickAlyson/Dissertacao/blob/main/images/capitulo3/modified_power.jpg?raw=true" />

In the figure bellow we display the result of this filtering algorithm for a beam image. We have the original image with the interference pattern (a), the notch filter generated (b) and the resulting image (c).

<img src="https://github.com/PatrickAlyson/Dissertacao/blob/main/images/capitulo3/notch.jpg?raw=true" />

## References
[1] CANNON, M.; LEHAR, A.; PRESTON, F. Background pattern removal by power
    spectral Altering. Appl. Opt., Optica Publishing Group, v. 22, n. 6, p. 777-779, Mar 1983.

[2] WELCH, P. The use of fast fourier transform for the estimation of power spectra: A
method based on time averaging over short, modified periodograms. IEEE Transactions on
Audio and Electroacoustics, v. 15, n. 2, p. 70-73, 1967.
