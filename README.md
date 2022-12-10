# MIST_Alloo23

This Python3 script has been written by Samantha J Alloo -  a PhD candidate at the University of Canterbury in New Zealand.

It was written to implement the newest theoretical approach of “Multimodal Intrinsic Speckle-Tracking (MIST)” which reconstructs the phase-shift, projected attenuation, and small-angle scattering (dark-field) signals of a given object when using a speckle-based phase-contrast X-ray imaging technique. When using this approach and script, please reference and give appropriate acknowledgement to this work published in XX: reference YY. 

The approach requires at least 4 sets of speckle-based imaging data to reconstruct the multimodal signals. Both reference-speckle (X-ray + detector + mask) and sample-reference-speckle (X-ray + detector + mask + sample) images are required to reconstruct such multimodal signals. 
The script first requires inputs of experimental parameters (ratio of the refractive index components for the object, sample-to-detector distance, detector pixel-size, and X-ray wavelength) and then imports the speckle intensity data in a for-loop. From here, a system of linear equations is solved using a Tikhonov regularised QR decomposition, from which the solutions are used to reconstruct:
-	A phase-object approximation of the object’s true dark-field signal 
-	The X-ray phase-shifts induced by the sample 
-	The object’s attenuation term 
-	A weakly-attenuating approximation of the object’s true dark-field signal 
