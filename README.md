This project implements a simulate-and-recover experiment to evaluate the consistency of the EZ diffusion model. The EZ diffusion model is a simplified version of the drift diffusion model, often used in cognitive science to analyze decision-making processes. The objective of this experiment is to determine whether the EZ diffusion model can reliably estimate its own parameters when applied to data it generates.

The simulation begins by randomly selecting parameters within predefined ranges: boundary separation (a) between 0.5 and 2.0, drift rate (v) between 0.5 and 2.0, and non-decision time (t) between 0.1 and 0.5. Using these parameters, synthetic reaction time (RT) and accuracy data are generated according to the forward equations of the EZ diffusion model. The generated data is then stored in results files for further processing. Once the data is simulated, the inverse equations of the EZ diffusion model are used to recover the original parameters from observed statistics, which include accuracy rate (R_obs), mean RT (M_obs), and RT variance (V_obs). This process is repeated 1000 times for each of the three sample sizes: N = 10, N = 40, and N = 4000. The recovered parameters are stored in separate files, allowing for a direct comparison between the true and estimated values. The accuracy of the recovered parameters is assessed by computing bias (the average difference between recovered and true parameters) and mean squared error (MSE), which measures estimation error.

The results of the experiment indicate that the EZ diffusion model is generally able to recover its own parameters, but with varying levels of accuracy depending on the sample size. When N is small, there is greater variability in the recovered parameters, leading to higher bias and error. As the sample size increases, bias and MSE decrease, indicating that larger datasets yield more precise parameter estimates. The results confirm that the EZ diffusion model is self-consistent but highlight the importance of using sufficiently large datasets to minimize estimation errors. Some numerical instability was observed at small N, requiring adjustments to the recovery process to improve stability.

This experiment demonstrates that the EZ diffusion model is capable of accurately estimating its own parameters, provided that the sample size is sufficiently large. The decreasing bias and MSE observed with increasing N indicate that parameter estimation improves with more data, reinforcing the importance of using large datasets when applying the model in cognitive research. However, small sample sizes introduce variability and potential bias, which researchers should account for when using the EZ diffusion model. This study highlights the need for careful consideration of sample size in parameter estimation and reinforces the broader importance of validation in scientific modeling. By refining numerical methods and ensuring robust data collection, we can enhance the reliability of cognitive models and improve their applicability in research.
